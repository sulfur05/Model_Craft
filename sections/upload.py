import streamlit as st
import pandas as pd
import numpy as np


MAX_FILE_SIZE_MB = 50


def _read_uploaded_file(uploaded_file, max_file_size_mb: int):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        st.warning(
            f"The file is about {file_size_mb:.1f} MB, which is larger than the recommended {max_file_size_mb} MB. "
            "Loading and analysis may be slow. Consider using a smaller sample of your data."
        )

    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Could not read the file. Error: {e}")
        return None


def _validate_and_preview_dataframe(df: pd.DataFrame):
    if df.empty:
        st.error("The uploaded file appears to be empty.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        st.error(
            "The dataset does not seem to contain numeric or categorical columns "
            "that can be used for machine learning."
        )
        return

    st.session_state["dataset"] = df
    st.session_state["numeric_columns"] = numeric_cols
    st.session_state["categorical_columns"] = categorical_cols

    st.success("Dataset loaded successfully! Here is a quick overview:")
    st.write(f"- Number of rows: {df.shape[0]}")
    st.write(f"- Number of columns: {df.shape[1]}")
    st.write("First few rows of your data:")
    st.dataframe(df.head())
    st.write("Column types:")
    st.write(df.dtypes.to_frame("dtype"))
    st.info(
        "In simple terms: each row is an example, and each column is a feature or variable. "
        "Next, you’ll choose which column you want to predict (the target)."
    )


def dataset_upload_section():
    with st.expander("1. Dataset Upload", expanded=True):
        st.subheader("Upload your dataset")

        st.write(
            """
            Upload a CSV or Excel file (up to about 50 MB).
            ModelCraft will load it, check that it looks valid, and show a quick preview.
            """
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="For now, only tabular data files are supported.",
        )

        if uploaded_file is not None:
            df = _read_uploaded_file(uploaded_file, MAX_FILE_SIZE_MB)
            if df is not None:
                _validate_and_preview_dataframe(df)
        else:
            st.info("Upload a CSV or Excel file to get started.")