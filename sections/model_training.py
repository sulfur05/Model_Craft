import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


def _ensure_preprocessed_data():
    if "preprocessor" not in st.session_state:
        st.info(
            "Run the 'Data Preprocessing' step first to configure preprocessing and split the data."
        )
        return None

    required_keys = ["X_train", "X_test", "y_train", "y_test", "feature_columns"]
    missing = [k for k in required_keys if k not in st.session_state]
    if missing:
        st.error(
            "Some preprocessing outputs are missing: " + ", ".join(missing) + ". "
            "Please re-run the preprocessing step."
        )
        return None

    return {
        "preprocessor": st.session_state["preprocessor"],
        "X_train": st.session_state["X_train"],
        "X_test": st.session_state["X_test"],
        "y_train": st.session_state["y_train"],
        "y_test": st.session_state["y_test"],
        "feature_columns": st.session_state["feature_columns"],
    }


def _get_model_options(task_type: str):
    if task_type == "regression":
        models = [
            "Linear Regression",
            "Ridge Regression",
            "Random Forest Regressor",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost Regressor")
    else:
        models = [
            "Logistic Regression",
            "Random Forest Classifier",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost Classifier")
    return models


def _build_model(task_type: str, model_name: str, params: dict):
    if task_type == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Ridge Regression":
            return Ridge(alpha=params.get("alpha", 1.0))
        if model_name == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        if model_name == "XGBoost Regressor" and HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )
    else:
        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 1.0),
                n_jobs=-1,
            )
        if model_name == "Random Forest Classifier":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        if model_name == "XGBoost Classifier" and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
    raise ValueError(f"Unsupported model: {model_name}")


def _train_and_evaluate(task_type: str, model_name: str, config, params: dict):
    preprocessor = config["preprocessor"]
    X_train = config["X_train"]
    X_test = config["X_test"]
    y_train = config["y_train"]
    y_test = config["y_test"]

    # Transform features using the preprocessing pipeline
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model = _build_model(task_type, model_name, params)
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)

    st.session_state["trained_model"] = model
    st.session_state["trained_model_name"] = model_name

    if task_type == "regression":
        _show_regression_results(y_test, y_pred)
    else:
        _show_classification_results(y_test, y_pred)


def _show_classification_results(y_true, y_pred):
    st.subheader("Evaluation metrics (classification)")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    st.write(f"Accuracy: **{acc:.3f}**")
    st.write(f"Precision (weighted): **{prec:.3f}**")
    st.write(f"Recall (weighted): **{rec:.3f}**")
    st.write(f"F1-score (weighted): **{f1:.3f}**")

    st.caption(
        "Accuracy is the overall fraction of correct predictions. "
        "Precision, recall, and F1 take class balance into account."
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)


def _show_regression_results(y_true, y_pred):
    st.subheader("Evaluation metrics (regression)")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"R² (explained variance): **{r2:.3f}**")
    st.write(f"MAE (mean absolute error): **{mae:.3f}**")
    st.write(f"RMSE (root mean squared error): **{rmse:.3f}**")

    st.caption(
        "R² measures how much of the variation in the target is explained by the model. "
        "Lower MAE/RMSE mean predictions are closer to the true values."
    )

    # Predicted vs actual scatter
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Predicted vs. true values")
    ax.legend()
    st.pyplot(fig)


def model_training_section():
    with st.expander("4. Model Training"):
        st.subheader("Choose a model and train it")

        config = _ensure_preprocessed_data()
        if config is None:
            return

        task_type = st.session_state.get("task_type", "classification")
        target_column = st.session_state.get("target_column", "target")

        st.write(
            f"You are solving a **{task_type}** problem to predict **{target_column}**. "
            "Below, choose a model and (optionally) adjust its settings."
        )

        model_options = _get_model_options(task_type)
        default_model_index = 0
        selected_model = st.selectbox("Model", options=model_options, index=default_model_index)

        params = {}
        st.markdown("**Model hyperparameters (basic)**")
        if task_type == "classification":
            if selected_model == "Logistic Regression":
                C = st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
                max_iter = st.slider("Max iterations", 100, 2000, 500, 100)
                params["C"] = C
                params["max_iter"] = max_iter
            elif selected_model in ["Random Forest Classifier", "XGBoost Classifier"]:
                n_estimators = st.slider("Number of trees", 50, 500, 200, 50)
                max_depth = st.slider("Max depth of trees (0 = unlimited)", 0, 30, 0, 1)
                params["n_estimators"] = n_estimators
                params["max_depth"] = max_depth or None
                if selected_model == "XGBoost Classifier":
                    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
                    params["learning_rate"] = learning_rate
        else:  # regression
            if selected_model == "Linear Regression":
                # No hyperparameters for basic LinearRegression
                st.caption("Linear Regression has no key hyperparameters for this simple setup.")
            elif selected_model == "Ridge Regression":
                alpha = st.slider("Regularization strength (alpha)", 0.01, 10.0, 1.0, 0.01)
                params["alpha"] = alpha
            elif selected_model in ["Random Forest Regressor", "XGBoost Regressor"]:
                n_estimators = st.slider("Number of trees", 50, 500, 200, 50)
                max_depth = st.slider("Max depth of trees (0 = unlimited)", 0, 30, 0, 1)
                params["n_estimators"] = n_estimators
                params["max_depth"] = max_depth or None
                if selected_model == "XGBoost Regressor":
                    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
                    params["learning_rate"] = learning_rate

        if st.button("Train model"):
            with st.spinner("Training model and evaluating on test data..."):
                _train_and_evaluate(task_type, selected_model, config, params)
            st.success("Training complete. See metrics and plots above.")
