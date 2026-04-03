import streamlit as st

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

with st.expander("1. Dataset Upload", expanded=True):
    st.write("Upload your CSV/Excel dataset here. (Coming soon)")

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