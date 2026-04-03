import streamlit as st

from sections.upload import dataset_upload_section
from sections.eda import eda_section
from sections.preprocessing import preprocessing_section
# from sections.model_training import model_training_section  # later
# from sections.explainability import explainability_section  # later
# from sections.export import export_section  # later
# from sections.advisor import advisor_section  # later


def main():
    st.title("ModelCraft – ML Workflow Assistant")
    st.write(
        """
        ModelCraft helps students and non-technical users run complete machine learning workflows
        on tabular datasets: from upload and exploration to training, explainability, and export.
        Use the sections below to move through the workflow step by step.
        """
    )

    dataset_upload_section()
    eda_section()
    preprocessing_section()
    # model_training_section()
    # explainability_section()
    # export_section()
    # advisor_section()


if __name__ == "__main__":
    st.set_page_config(
        page_title="ModelCraft – ML Workflow Assistant",
        layout="wide",
    )
    main()