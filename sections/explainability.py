# sections/explainability.py
# Explainability UI + helpers (SHAP) with inline comments explaining each line/block.

import streamlit as st                       # Streamlit UI primitives and caching/session state
import numpy as np                           # Numeric computations
import pandas as pd                          # DataFrame helpers
import matplotlib.pyplot as plt              # For plotting SHAP/matplotlib figures
import hashlib                               # For creating cache keys (md5)
import cloudpickle                           # Robust serializer for model/preprocessor fingerprinting
import shap                                  # SHAP explainability library
import joblib                                # (optional) model persistence helpers
from typing import Tuple, Any                # Type hints for readability

# UI / algorithm defaults
DEFAULT_SAMPLE = 500                         # Default number of rows to sample when computing SHAP
SHAP_KERNEL_BACKGROUND = 100                # Background sample size for KernelExplainer (keep small)

def _fingerprint(obj: Any) -> str:
    """
    Create a stable fingerprint for any Python object (model/preprocessor).
    We use cloudpickle to serialize (works for sklearn pipelines/models), then md5 digest.
    This fingerprint is used as part of cache keys so cached explainers/values map to the right model.
    """
    try:
        # Serialize object to bytes; cloudpickle handles many Python objects that pickle can't.
        b = cloudpickle.dumps(obj)
    except Exception:
        # If serialization fails, fallback to the string representation (less stable but safe).
        b = repr(obj).encode()
    # Return an MD5 hex digest of the bytes (short fixed-length key)
    return hashlib.md5(b).hexdigest()

@st.cache_resource
def _build_explainer_cached(_model, X_background: pd.DataFrame, model_fp: str):
    try:
        expl = shap.Explainer(_model, X_background, feature_names=X_background.columns)
        return expl
    except Exception:
        bg = X_background.sample(min(len(X_background), SHAP_KERNEL_BACKGROUND), random_state=0)
        expl = shap.KernelExplainer(
            lambda x: _model.predict_proba(x) if hasattr(_model, "predict_proba") else _model.predict(x),
            bg,
        )
        return expl

@st.cache_data(show_spinner=False)
def _compute_shap_values_cached(explainer, X: pd.DataFrame, model_fp: str, sample_n: int):
    """
    Compute SHAP values for a sample of X and cache the result.
    - We sample min(len(X), sample_n) rows to keep computation manageable.
    - Returns a SHAP Explanation object (`shap_values`) and the sampled DataFrame (`Xs`).
    - Decorated with `@st.cache_data` to reuse results across reruns when inputs don't change.
    """
    # Subsample rows deterministically for reproducibility
    X_sample_raw = X.sample(min(len(X), sample_n), random_state=0)
    # X_sample_raw = Xn.sample()
    X_sample = pd.DataFrame(st.session_state["preprocessor"].transform(X_sample_raw), columns=st.session_state["feature_columns"])

    # Compute SHAP values for the sample (explainer can return an Explanation object)

    shap_values = explainer(X_sample)
    Xs = X_sample
    return shap_values, Xs

def _ensure_trained_model():
    """
    Guard helper: ensure a model was trained and stored in Streamlit session state.
    If not present, show an informational message and return None.
    """
    if "trained_model" not in st.session_state:
        st.info("Train a model first (step 4) to use Explainability.")
        return None
    return st.session_state["trained_model"]

def explainability_section():
    """
    Streamlit UI: configure and compute SHAP explanations, show a short plain-language summary,
    and render common SHAP plots (summary bar, beeswarm, dependence, force).
    Explanations and heavy objects are cached to avoid recomputation on every UI interaction.
    """
    with st.expander("5. Explainability (SHAP)"):
        # Guard: require a trained model
        model = _ensure_trained_model()
        if model is None:
            return

        # Guard: require preprocessing pipeline and training data
        if "preprocessor" not in st.session_state:
            st.info("Preprocessing pipeline not found — run preprocessing first.")
            return

        # Grab training features and feature names (feature_columns saved in preprocessing step)
        X_train = st.session_state["X_train"]
        feature_columns = st.session_state.get("feature_columns", X_train.columns.tolist())

        # Small instructions and controls for the user
        st.write("Configure SHAP and compute explanations (sample-based for performance).")
        # Slider to control how many rows are used for SHAP (tradeoff: speed vs fidelity)
        sample_n = st.slider("Sample size for SHAP (rows)", 50, DEFAULT_SAMPLE, min(200, DEFAULT_SAMPLE), step=50)
        # Button to force recompute (bypass cache)
        refresh = st.button("(Re)compute SHAP")

        # Create a fingerprint key combining model and preprocessor fingerprints
        model_fp = _fingerprint(st.session_state["trained_model"]) + "_" + _fingerprint(st.session_state["preprocessor"])

        # Prepare a background dataset (raw) that we will transform with the preprocessor
        preprocessor = st.session_state["preprocessor"]
        # Choose a background raw sample: min(len(X_train), max(sample_n, 200))
        X_background_raw = X_train.sample(min(len(X_train), max(sample_n, 200)), random_state=0)
        try:
            # Attempt to transform the raw background rows and keep feature names
            X_background = pd.DataFrame(preprocessor.transform(X_background_raw), columns=feature_columns)
        except Exception:
            # If transformer returns a numpy array without column names, coerce to DataFrame without names
            X_background = pd.DataFrame(preprocessor.transform(X_background_raw))

        # Build or fetch cached explainer for this model + background
        explainer = _build_explainer_cached(model, X_background, model_fp)

        # Compute (or fetch cached) SHAP values for a sample of X_train
        if refresh:
            # Force recompute by calling the underlying function (bypass cache) and use a fresh key
            shap_vals, Xs = _compute_shap_values_cached.__wrapped__(explainer, X_train, model_fp + "_fresh", sample_n)
            # Store results in session_state for immediate use
            st.session_state["_shap_values"] = shap_vals
            st.session_state["_shap_Xs"] = Xs
        else:
            try:
                # Normal cached path: may return cached results if inputs match previous call
                shap_vals, Xs = _compute_shap_values_cached(explainer, X_train, model_fp, sample_n)
            except Exception:
                # If compute fails (e.g., not cached yet), instruct user to press refresh
                st.warning("SHAP computation failed or not cached yet; click '(Re)compute SHAP' to run.")
                return

        # Save shap values + sample into session state so other parts (advisor/export) can reuse them
        st.session_state["_shap_values"] = shap_vals
        st.session_state["_shap_Xs"] = Xs

        # Create a short human-friendly summary of top influencing features (plain-language bullets)
        try:
            # `shap_vals.values` is typically array-like shape (n_samples, n_features). Compute mean absolute contribution per feature.
            abs_mean = np.mean(np.abs(shap_vals.values), axis=0)
            # Get indices of top features by importance
            top_idx = np.argsort(abs_mean)[::-1][:5]
            # Map indices back to feature names
            top_features = [shap_vals.feature_names[i] for i in top_idx]
            top_scores = abs_mean[top_idx]
            # Concise summary string with top features and their mean |SHAP| scores
            summary_text = ", ".join([f"{f} ({s:.3f})" for f, s in zip(top_features, top_scores)])
            st.markdown("**Top influencing features (SHAP mean |value|):**")
            st.write(summary_text)

            # Produce 1–3 plain-language intuition lines for the advisor/UI:
            plain_lines = []
            for i, idx in enumerate(top_idx[:3]):
                f = shap_vals.feature_names[idx]
                # Determine the average sign of the SHAP values for this feature (positive/negative effect)
                mean_sh = np.sign(np.mean(shap_vals.values[:, idx]))
                dir_text = "increases" if mean_sh > 0 else "decreases or is associated with lower"
                plain_lines.append(f"- Higher values of **{f}** tend to {dir_text} the model's predicted score.")
            st.markdown("**Quick intuition:**")
            for L in plain_lines:
                st.write(L)

            # Store the plain-language summary in session_state for reuse by the advisor
            st.session_state["_shap_summary_plain"] = plain_lines
        except Exception:
            # If anything in the summary generation fails, show a short info and continue
            st.info("Could not produce top-feature summary for this model.")

        st.markdown("---")

        # Provide a selector for which SHAP plot to render
        plot_type = st.selectbox("Select SHAP plot", ["Summary (bar)", "Beeswarm", "Dependence", "Force plot"])

        if plot_type == "Summary (bar)":
            # Summary bar plot: average absolute SHAP value per feature
            fig = plt.figure(figsize=(6, max(3, len(shap_vals.feature_names) * 0.2)))
            shap.plots.bar(shap_vals, max_display=20, show=False)  # SHAP uses matplotlib/seaborn under the hood
            st.pyplot(fig)

        elif plot_type == "Beeswarm":
            # Beeswarm plot: distribution of SHAP values per feature across samples
            fig = plt.figure(figsize=(7, 5))
            shap.plots.beeswarm(shap_vals, max_display=40, show=False)
            st.pyplot(fig)

        elif plot_type == "Dependence":
            # Dependence plot: how SHAP value for a feature varies vs feature value (and other interacting features)
            feat = st.selectbox("Feature for dependence plot", shap_vals.feature_names)
            plt.figure(figsize=(6, 4))
            # shap.dependence_plot accepts raw arrays/values; we pass the computed values + the sampled Xs
            shap.dependence_plot(feat, shap_vals.values, Xs, show=False)
            st.pyplot(plt.gcf())

        elif plot_type == "Force plot":
            # Force plot: local explanation for a single example (interactive/HTML preferred)
            idx = st.slider("Example index (from sample)", 0, len(shap_vals.values) - 1, 0)
            try:
                # shap.plots.force can return HTML/JS interactive output; render using Streamlit components
                html = shap.plots.force(shap_vals[idx], matplotlib=False)
                st.components.v1.html(html, height=350)
            except Exception:
                # If HTML rendering isn't supported, advise alternative plots
                st.warning("Force plot rendering not supported in this environment. Use Summary/Beeswarm instead.")

        # Small note to the user about performance/limits
        st.caption("Note: SHAP computation can be slow; reduce sample size to speed up.")