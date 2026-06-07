import os
import textwrap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _mentor_content():

    if "trained_model" in st.session_state:
        return (
            "Your model has finished training.",
            "I'd explore Explainability next to understand what influenced predictions.",
            "Feature importance often reveals surprising patterns in your data."
        )

    if "preprocessor" in st.session_state:
        return (
            "Your data is ready for machine learning.",
            "I'd train a model next and use it as a baseline.",
            "Don't worry about finding the perfect model immediately."
        )

    if st.session_state.get("eda_complete", False):
        return (
            "Your dataset has been explored.",
            "I'd configure preprocessing next.",
            "Good preprocessing often improves model quality more than changing algorithms."
        )

    if "dataset" in st.session_state:
        return (
            "Your dataset is uploaded and ready.",
            "I'd run EDA next before training anything.",
            "Understanding your data early prevents many common ML mistakes."
        )

    return (
        "Welcome to ModelCraft.",
        "Start by uploading a dataset.",
        "Most ML projects begin with understanding the data, not choosing a model."
    )


def _call_llm(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "No AI backend configured. "
            "Add GROQ_API_KEY to enable mentor responses."
        )

    import requests

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def advisor_panel():

        observation, recommendation, note = _mentor_content()
    
        st.markdown("## 🧠 AI Mentor")
    
        st.info(
            f"""
    {observation}

    {recommendation}
    """
        )

        st.caption(f"💡 {note}")

        st.divider()

        if "mentor_chat_open" not in st.session_state:
            st.session_state["mentor_chat_open"] = False

        if not st.session_state["mentor_chat_open"]:

            if st.button(
                "💬 Ask a Question",
                use_container_width=True
            ):
                st.session_state["mentor_chat_open"] = True
                st.rerun()

            return

        with st.container(border=True):

            question = st.text_area(
                "",
                placeholder="Why is my accuracy low?",
                label_visibility="collapsed",
                height=100,
            )

            col1, col2 = st.columns(2)

            with col1:

                send = st.button(
                    "Send",
                    use_container_width=True,
                )

            with col2:

                close = st.button(
                    "Close",
                    use_container_width=True,
                )

            if close:
                st.session_state["mentor_chat_open"] = False
                st.rerun()

            if send and question.strip():

                prompt = f"""
    You are ModelCraft AI Mentor.

    The user is a beginner learning machine learning.

    Question:
    {question}

    Rules:
    - Keep under 100 words
    - Use simple language
    - Give practical advice
    - Avoid jargon
    """

                with st.spinner("Thinking..."):
                    answer = _call_llm(prompt)

                st.success(answer)