import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ModelCraft – ML Workflow Assistant",
    layout="wide",
)

st.title("ModelCraft – ML Workflow Assistant")
st.write(
    """
ModelCraft helps students and non-technical users run complete machine learning workflows
on tabular datasets: from upload and exploration to training, explainability, and export.
Use the sections below to move through the workflow step by step.
"""
)

MAX_FILE_SIZE_MB = 50

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
        # Soft size check
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.warning(
                f"The file is about {file_size_mb:.1f} MB, which is larger than the recommended {MAX_FILE_SIZE_MB} MB. "
                "Loading and analysis may be slow. Consider using a smaller sample of your data."
            )

        # Try to read the file into pandas
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                df = None
        except Exception as e:
            st.error(f"Could not read the file. Error: {e}")
            df = None

        if df is not None:
            # Basic validation: non-empty and at least some usable columns
            if df.empty:
                st.error("The uploaded file appears to be empty.")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

                if not numeric_cols and not categorical_cols:
                    st.error(
                        "The dataset does not seem to contain numeric or categorical columns "
                        "that can be used for machine learning."
                    )
                else:
                    # Save to session state for other sections
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
    else:
        st.info("Upload a CSV or Excel file to get started.")

with st.expander("2. Exploratory Data Analysis (EDA)"):
    st.write("Visualize your data: distributions, correlations, and missing values. (Coming soon)")

with st.expander("3. Data Preprocessing"):
    st.write("Configure how to clean and transform your data. (Coming soon)")

with st.expander("4. Model Training"):
    st.write("Choose a model, set basic parameters, and train it. (Coming soon)")

with st.expander("5. Explainability (SHAP)"):
    st.write("Understand which features influence your model’s predictions. (Coming soon)")

with st.expander("6. Export & Report"):
    st.write("Save your trained model and generate a simple report. (Coming soon)")

with st.expander("7. AI Advisor"):
    st.write("Chat with an assistant that guides you through the workflow. (Coming soon)")