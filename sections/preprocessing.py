import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


def _ensure_dataset_and_target():
    if "dataset" not in st.session_state:
        st.info("Upload a dataset in step 1 (Dataset Upload) first.")
        return None, None, None, None

    if "target_column" not in st.session_state:
        st.info("Choose a target column in the upload step before preprocessing.")
        return None, None, None, None

    df: pd.DataFrame = st.session_state["dataset"]
    target_column: str = st.session_state["target_column"]
    numeric_cols = st.session_state.get("numeric_columns", [])
    categorical_cols = st.session_state.get("categorical_columns", [])

    if target_column not in df.columns:
        st.error("The selected target column is not in the dataset.")
        return None, None, None, None

    return df, target_column, numeric_cols, categorical_cols


def _build_numeric_transformer(strategy: str, scaler_name: str):
    if strategy == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif strategy == "Median":
        imputer = SimpleImputer(strategy="median")
    elif strategy == "Most frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    else:  # Constant 0
        imputer = SimpleImputer(strategy="constant", fill_value=0)

    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )


def _build_categorical_transformer(strategy: str, encoder_name: str):
    if strategy == "Most frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    else:  # Constant "Missing"
        imputer = SimpleImputer(strategy="constant", fill_value="Missing")

    if encoder_name == "One-hot (recommended)":
        encoder = OneHotEncoder(handle_unknown="ignore")
    elif encoder_name == "Label / ordinal encoding":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        # No encoding: pass through as-is
        encoder = "passthrough"

    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("encoder", encoder),
        ]
    )


def preprocessing_section():
    with st.expander("3. Data Preprocessing"):
        st.subheader("Configure how to clean and split your data")

        df, target_column, numeric_cols, categorical_cols = _ensure_dataset_and_target()
        if df is None:
            return

        st.write(
            "Here you choose how to handle missing values, encode categories, "
            "scale numeric features, and split your data into train and test sets."
        )

        # Optionally drop rows with missing target
        drop_missing_target = st.checkbox(
            "Drop rows where the target value is missing (recommended)",
            value=True,
        )

        if drop_missing_target:
            initial_rows = len(df)
            df = df.dropna(subset=[target_column])
            dropped = initial_rows - len(df)
            if dropped > 0:
                st.caption(f"Dropped {dropped} rows with missing target values.")

        feature_columns = [c for c in df.columns if c != target_column]

        st.markdown("**Global strategies for missing values**")
        num_missing_strategy = st.selectbox(
            "Numeric columns:",
            options=["Mean", "Median", "Most frequent", "Constant 0"],
            index=0,
        )
        cat_missing_strategy = st.selectbox(
            "Categorical columns:",
            options=["Most frequent", "Constant 'Missing'"],
            index=0,
        )

        st.markdown("**Encoding for categorical variables**")
        encoder_name = st.selectbox(
            "Choose encoder:",
            options=[
                "One-hot (recommended)",
                "Label / ordinal encoding",
                "No encoding (keep as text)",
            ],
            index=0,
            help=(
                "One-hot encoding creates separate 0/1 columns for each category. "
                "Label/ordinal encoding turns categories into numbers (1, 2, 3...)."
            ),
        )

        st.markdown("**Scaling for numeric variables**")
        scaler_name = st.selectbox(
            "Numeric scaler:",
            options=["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            index=0,
            help=(
                "Scaling can help some models by putting features on a similar scale. "
                "It is most useful for distance-based models and linear models."
            ),
        )

        st.markdown("**Train / test split**")
        test_size = st.slider(
            "Test set size (fraction of data)",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
        )

        task_type = st.session_state.get("task_type", "classification")
        stratify = False
        if task_type == "classification":
            stratify = st.checkbox(
                "Stratify split by target (keep class balance)",
                value=True,
            )

        if st.button("Apply preprocessing and split data"):
            _apply_preprocessing_and_split(
                df,
                target_column,
                feature_columns,
                numeric_cols,
                categorical_cols,
                num_missing_strategy,
                cat_missing_strategy,
                encoder_name,
                scaler_name,
                test_size,
                stratify,
                task_type,
            )


def _apply_preprocessing_and_split(
    df: pd.DataFrame,
    target_column: str,
    feature_columns,
    numeric_cols,
    categorical_cols,
    num_missing_strategy: str,
    cat_missing_strategy: str,
    encoder_name: str,
    scaler_name: str,
    test_size: float,
    stratify: bool,
    task_type: str,
):
    X = df[feature_columns]
    y = df[target_column]

    # Ensure we only use feature columns that actually exist after type detection
    num_features = [c for c in numeric_cols if c in feature_columns]
    cat_features = [c for c in categorical_cols if c in feature_columns]

    numeric_transformer = _build_numeric_transformer(num_missing_strategy, scaler_name)
    categorical_transformer = _build_categorical_transformer(cat_missing_strategy, encoder_name)

    transformers = []
    if num_features:
        transformers.append(("num", numeric_transformer, num_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    if not transformers:
        st.error("No valid numeric or categorical feature columns available for preprocessing.")
        return

    preprocessor = ColumnTransformer(transformers=transformers)

    # Train-test split
    stratify_arg = y if (task_type == "classification" and stratify) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_arg,
    )

    # Fit preprocessor on training data only
    preprocessor.fit(X_train)

    st.session_state["preprocessor"] = preprocessor
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["feature_columns"] = feature_columns

    st.success("Preprocessing configured and data split into train and test sets.")
    st.write(f"Training rows: {len(X_train)}")
    st.write(f"Test rows: {len(X_test)}")

    st.info(
        "The preprocessing pipeline and the train/test split are now saved. "
        "The model training step will reuse this configuration so that the same "
        "transformations are always applied."
    )
