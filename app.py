import streamlit as st
from dotenv import load_dotenv

from sections.advisor import advisor_panel
from sections.eda import eda_section
from sections.explainability_test import explainability_section
from sections.export import export_section
from sections.model_training import model_training_section
from sections.preprocessing import preprocessing_section
from sections.upload import dataset_upload_section

load_dotenv()


def load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as file:
        css = file.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    st.title("ModelCraft – ML Workflow Assistant")
    st.write(
        """
        ModelCraft helps students and non-technical users run complete machine learning workflows
        on tabular datasets: from upload and exploration to training, explainability, and export.
        Use the sections below to move through the workflow step by step.
        """
    )

    load_css("style.css")

    col_main, col_advisor = st.columns([3, 1])

    with col_main:
        dataset_upload_section()
        eda_section()
        preprocessing_section()
        model_training_section()
        explainability_section()
        export_section()

    with col_advisor:
        advisor_panel()


if __name__ == "__main__":
    st.set_page_config(
        page_title="ModelCraft – ML Workflow Assistant",
        layout="wide",
    )
    main()