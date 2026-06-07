import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap as sp

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def _is_tree_model(model):
    tree_models = (RandomForestClassifier, RandomForestRegressor)
    if HAS_XGBOOST:
        tree_models = tree_models + (XGBClassifier, XGBRegressor)
    return isinstance(model, tree_models)


def _is_linear_model(model):
    return isinstance(model, (LinearRegression, Ridge, LogisticRegression))


def _is_classifier_model(model):
    classifier_models = (LogisticRegression, RandomForestClassifier)
    if HAS_XGBOOST:
        classifier_models = classifier_models + (XGBClassifier,)
    return isinstance(model, classifier_models)


def _prepare_shap_input(preprocessor, X_train):
    X_train_proc = preprocessor.transform(X_train)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    if X_train_proc.ndim == 1:
        X_train_proc = X_train_proc.reshape(-1, 1)

    if len(feature_names) != X_train_proc.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    return X_train_proc


def _select_positive_class(shap_values):
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        return shap_values[..., 1]
    return shap_values


def explainability_section():
    with st.expander("5. Explainability section"):
        if "trained_model" not in st.session_state:
            st.info("No trained model in session_state")
            return

        if "preprocessor" not in st.session_state or "X_train" not in st.session_state:
            st.info("Run preprocessing before using explainability.")
            return

        model = st.session_state["trained_model"]
        st.write("Model type:", type(model).__name__)

        X_train = st.session_state["X_train"]
        preprocessor = st.session_state["preprocessor"]
        X_train_proc = _prepare_shap_input(preprocessor, X_train)

        if _is_linear_model(model):
            explainer = sp.LinearExplainer(model, X_train_proc)
            shap_values = explainer(X_train_proc)

        elif _is_tree_model(model):
            explainer = sp.TreeExplainer(model)
            shap_values = explainer(X_train_proc)

        else:
            explainer = sp.Explainer(model, X_train_proc)
            shap_values = explainer(X_train_proc)

        if _is_classifier_model(model):
            shap_values = _select_positive_class(shap_values)

        # 🔴 CHANGE 4: Organize SHAP in tabs
        tab1, tab2 = st.tabs(["📊 Simple Explanation", "🔍 Detailed Analysis"])
        
        with tab1:
            st.subheader("Top Features Affecting Predictions")
            st.markdown("These are the most important features influencing your model's decisions.")
            fig1 = plt.figure(figsize=(8, 5))
            sp.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig1, clear_figure=True)

        with tab2:
            st.subheader("Global feature importance")
            fig1 = plt.figure(figsize=(8, 5))
            sp.plots.bar(shap_values, max_display=15, show=False)
            st.pyplot(fig1, clear_figure=True)

            st.subheader("Feature impact distribution")
            st.markdown("Shows how different feature values impact predictions.")
            fig2 = plt.figure(figsize=(8, 5))
            sp.plots.beeswarm(shap_values, max_display=15, show=False)
            st.pyplot(fig2, clear_figure=True)


explainability_section()