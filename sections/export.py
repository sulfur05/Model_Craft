from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from utils.reporting import create_pdf_report_bytes
import numpy as np

from sections.model_management import (
    load_model_bundle,
    list_saved_models,
    predict_with_bundle,
    save_model_bundle,
    validate_model_bundle,
    build_model_bundle,
)

from utils.reporting import create_pdf_report_bytes

def _display_bundle_summary(bundle: dict) -> None:
    model_name = bundle.get("trained_model_name", "Unknown")
    version = bundle.get("version", "Unknown")
    task_type = bundle.get("task_type", "Unknown")
    target_column = bundle.get("target_column", "Unknown")
    created_at = bundle.get("created_at", "Unknown")
    dataset_shape = bundle.get("dataset_shape", "Unknown")
    metrics = bundle.get("metrics", {})

    st.write(f"**Model:** {model_name}")
    st.write(f"**Version:** {version}")
    st.write(f"**Task type:** {task_type}")
    st.write(f"**Target column:** {target_column}")
    st.write(f"**Created at:** {created_at}")
    st.write(f"**Dataset shape:** {dataset_shape}")

    if metrics:
        st.subheader("Saved metrics")
        st.json(metrics)

    if bundle.get("comparison_results") is not None:
        st.subheader("Model comparison results")
        try:
            st.dataframe(bundle["comparison_results"])
        except Exception:
            st.write(bundle["comparison_results"])

def _gather_full_report_artifacts(bundle: dict):
    eda_figs = []
    shap_figs = []
    ds_text = None
    preprocessing_summary = None
    model_params = None
    model_metrics = None
    comparison_df = bundle.get("comparison_results")
    confusion_fig = None
    pred_vs_actual_fig = None
    sample_predictions = None

    df = st.session_state.get("dataset")
    if df is not None:
        ds_text = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}. Columns: {list(df.columns)[:15]}{' ...' if df.shape[1]>15 else ''}."
        # missing values plot
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            fig, ax = plt.subplots(figsize=(6,3))
            missing.plot(kind="bar", ax=ax)
            ax.set_title("Missing values by column")
            eda_figs.append(fig)
        # a few histograms and boxplots (limit to 3)
        numeric = st.session_state.get("numeric_columns", [])[:3]
        for col in numeric:
            fig, ax = plt.subplots(figsize=(5,3))
            df[col].dropna().hist(ax=ax, bins=30)
            ax.set_title(f"Histogram: {col}")
            eda_figs.append(fig)
            # boxplot
            fig2, ax2 = plt.subplots(figsize=(5,2))
            ax2.boxplot(df[col].dropna())
            ax2.set_title(f"Boxplot: {col}")
            eda_figs.append(fig2)
        # correlation heatmap (small)
        numeric_all = st.session_state.get("numeric_columns", [])
        if len(numeric_all) >= 2:
            sample = df[numeric_all].sample(min(len(df), 2000), random_state=0)
            corr = sample.corr()
            fig, ax = plt.subplots(figsize=(6,6))
            import seaborn as sns
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation heatmap")
            eda_figs.append(fig)

    # preprocessing summary
    preprocessor = st.session_state.get("preprocessor")
    if preprocessor is not None:
        try:
            preprocessing_summary = str(preprocessor)
        except Exception:
            preprocessing_summary = "Preprocessor present; details unavailable."

    # model params and metrics
    model = st.session_state.get("trained_model")
    if model is not None:
        if hasattr(model, "get_params"):
            try:
                model_params = model.get_params()
            except Exception:
                model_params = None
    model_metrics = st.session_state.get("trained_model_metrics") or bundle.get("metrics")

    # diagnostic plots from training step if available in session state (or recompute)
    # confusion matrix or pred vs actual may not be saved - try to generate small versions if possible
    last_y_test = st.session_state.get("y_test")
    last_y_pred = None
    try:
        if "trained_model" in st.session_state and last_y_test is not None:
            model = st.session_state["trained_model"]
            X_test = st.session_state.get("X_test")
            pre = st.session_state.get("preprocessor")
            X_test_proc = pre.transform(X_test)
            last_y_pred = model.predict(X_test_proc)
            # confusion for classification
            if st.session_state.get("task_type") == "classification":
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(last_y_test, last_y_pred)
                fig, ax = plt.subplots(figsize=(4,3))
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion matrix (test set)")
                confusion_fig = fig
            else:
                # regression pred vs actual
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(last_y_test, last_y_pred, alpha=0.6)
                ax.plot([min(last_y_test), max(last_y_test)], [min(last_y_test), max(last_y_test)], "r--")
                ax.set_title("Predicted vs Actual (test set)")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                pred_vs_actual_fig = fig
    except Exception:
        pass

    # SHAP figs (try to reuse session shap values)
    shap_vals = st.session_state.get("_shap_values")
    if shap_vals is not None:
        try:
            fig = plt.figure(figsize=(7,4))
            shap.plots.bar(shap_vals, max_display=15, show=False)
            shap_figs.append(fig)
        except Exception:
            pass

    # Sample predictions: take a few rows from X_test if available
    if last_y_pred is not None and "X_test" in st.session_state:
        X_test = st.session_state["X_test"].copy()
        X_test = X_test.reset_index(drop=True)
        preds = list(last_y_pred)[:10]
        sample_predictions = X_test.head(len(preds)).copy()
        sample_predictions["prediction"] = preds

    return {
        "eda_figs": eda_figs,
        "shap_figs": shap_figs,
        "dataset_text": ds_text,
        "preprocessing_summary": preprocessing_summary,
        "model_params": model_params,
        "model_metrics": model_metrics,
        "comparison_df": comparison_df,
        "confusion_fig": confusion_fig,
        "pred_vs_actual_fig": pred_vs_actual_fig,
        "sample_predictions": sample_predictions,
    }

def _save_current_model_ui() -> None:
    st.subheader("Save current model")
    if "trained_model" not in st.session_state:
        st.info("Train a model first before exporting it.")
        return

    default_name = st.session_state.get("trained_model_name", "trained_model")
    model_name = st.text_input("Model name", value=default_name)
    version = st.text_input("Version (optional)", value="")

    if st.button("Save model bundle"):
        try:
            with st.spinner("Saving model bundle..."):
                result = save_model_bundle(model_name=model_name, version=version.strip() or None)
            bundle_path = result["bundle_path"]
            bundle = result["bundle"]
            st.success(f"Model saved to: {bundle_path}")
            st.session_state["last_saved_bundle_path"] = str(bundle_path)
            st.session_state["last_saved_bundle"] = bundle
            try:
                bundle_bytes = bundle_path.read_bytes()
                st.download_button(
                    label="Download model bundle",
                    data=bundle_bytes,
                    file_name=bundle_path.name,
                    mime="application/octet-stream",
                )
            except Exception:
                st.warning("Saved on disk, but download button unavailable.")
        except Exception as exc:
            st.error(f"Could not save model bundle: {exc}")

    st.markdown("---")
    st.write("Generate a full PDF report (dataset, EDA, model, SHAP)")

    if st.button("Generate PDF report"):
        try:
            with st.spinner("Generating PDF report..."):
                bundle = st.session_state.get("last_saved_bundle") or   build_model_bundle()
                artifacts = _gather_full_report_artifacts(bundle)
                pdf_bytes = create_pdf_report_bytes(
                    bundle=bundle,
                    dataset_summary_text=artifacts["dataset_text"],
                    eda_figs=artifacts["eda_figs"],
                    shap_figs=artifacts["shap_figs"],
                    preprocessing_summary=artifacts["preprocessing_summary"],
                    model_params=artifacts["model_params"],
                    model_metrics=artifacts["model_metrics"],
                    comparison_df=artifacts["comparison_df"],
                    confusion_fig=artifacts["confusion_fig"],
                    pred_vs_actual_fig=artifacts["pred_vs_actual_fig"],
                    sample_predictions=artifacts["sample_predictions"],
                )
                st.download_button("Download full PDF report", data=pdf_bytes,  file_name="modelcraft_report.pdf", mime="application/pdf")
        except Exception as exc:
            st.error(f"Could not generate PDF: {exc}")
            
def _load_saved_bundle_ui() -> None:
    st.subheader("Load existing model bundle")
    uploaded_bundle = st.file_uploader("Upload a saved model bundle (.joblib or .pkl)", type=["joblib","pkl"])
    if uploaded_bundle is not None:
        if st.button("Load uploaded bundle"):
            try:
                with st.spinner("Loading bundle..."):
                    bundle = load_model_bundle(uploaded_bundle, persist_to_session=True)
                st.session_state["loaded_bundle"] = bundle
                st.success("Model bundle loaded successfully.")
                _display_bundle_summary(bundle)
            except Exception as exc:
                st.error(f"Could not load bundle: {exc}")

    st.markdown("---")
    st.write("Saved model registry")
    try:
        registry_df = list_saved_models()
        if registry_df.empty:
            st.info("No saved models found in the exports folder.")
        else:
            cols_to_show = ["trained_model_name","version","created_at","task_type","target_column","feature_count","bundle_path"]
            st.dataframe(registry_df[cols_to_show])
            selected_index = st.selectbox(
                "Select a saved model to inspect",
                options=list(range(len(registry_df))),
                format_func=lambda i: f"{registry_df.iloc[i]['trained_model_name']} | {registry_df.iloc[i]['version']} | {registry_df.iloc[i]['created_at']}"
            )
            if st.button("Load selected model from registry"):
                try:
                    bundle_path = Path(registry_df.iloc[selected_index]["bundle_path"])
                    bundle = load_model_bundle(bundle_path, persist_to_session=True)
                    st.session_state["loaded_bundle"] = bundle
                    st.success("Selected bundle loaded successfully.")
                    _display_bundle_summary(bundle)
                except Exception as exc:
                    st.error(f"Could not load selected bundle: {exc}")
    except Exception as exc:
        st.warning(f"Could not read saved model registry: {exc}")

def _batch_prediction_ui() -> None:
    st.subheader("Batch prediction")
    if "trained_model" not in st.session_state or "preprocessor" not in st.session_state:
        st.info("Load a saved bundle or train a model first.")
        return

    bundle = {
        "model": st.session_state.get("trained_model"),
        "preprocessor": st.session_state.get("preprocessor"),
        "feature_columns": st.session_state.get("feature_columns", []),
        "target_column": st.session_state.get("target_column"),
        "task_type": st.session_state.get("task_type", "classification"),
        "trained_model_name": st.session_state.get("trained_model_name", "trained_model"),
        "created_at": st.session_state.get("created_at", ""),
        "version": st.session_state.get("version", ""),
        "dataset_shape": st.session_state.get("dataset_shape"),
        "metrics": st.session_state.get("trained_model_metrics", {}),
        "library_versions": st.session_state.get("library_versions", {}),
    }

    try:
        validate_model_bundle(bundle)
    except Exception:
        pass

    uploaded_csv = st.file_uploader("Upload CSV for prediction", type=["csv"], key="batch_csv_uploader")
    include_proba = st.checkbox("Include class probabilities when available", value=True)
    if uploaded_csv is None:
        return

    try:
        input_df = pd.read_csv(uploaded_csv)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    st.write("Preview of uploaded data")
    st.dataframe(input_df.head())

    expected_columns = st.session_state.get("feature_columns", [])
    if expected_columns:
        missing = [col for col in expected_columns if col not in input_df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            return

    if st.button("Run batch prediction"):
        try:
            with st.spinner("Generating predictions..."):
                result_df = predict_with_bundle(bundle=bundle, input_df=input_df, include_proba=include_proba)
            st.session_state["last_prediction_result"] = result_df
            st.success("Predictions generated successfully.")
            st.dataframe(result_df.head())
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")

def export_section() -> None:
    with st.expander("6. Model Export & Management"):
        st.write("Save the trained model together with its preprocessing pipeline, reload old versions, and run predictions on new CSV files.")
        tab_save, tab_load, tab_batch = st.tabs(["Save current model", "Load model bundle", "Batch prediction"])
        with tab_save:
            _save_current_model_ui()
        with tab_load:
            _load_saved_bundle_ui()
        with tab_batch:
            _batch_prediction_ui()