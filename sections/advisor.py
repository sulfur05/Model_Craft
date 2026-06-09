import os
import textwrap
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _mentor_content():

    if "trained_model" in st.session_state:
        return (
            "🎉 Your model has finished training.",
            "I'd explore Explainability next.",
            "Feature importance often reveals surprising patterns in your data."
        )

    if "preprocessor" in st.session_state:
        return (
            "✅ Your data is ready for machine learning.",
            "I'd train a model next.",
            "Start with a baseline before worrying about tuning."
        )

    if st.session_state.get("eda_complete", False):
        return (
            "📊 Your dataset has been explored.",
            "I'd configure preprocessing next.",
            "Good preprocessing often improves results more than switching algorithms."
        )

    if "dataset" in st.session_state:
        return (
            "📁 Your dataset is uploaded and ready.",
            "I'd run EDA next.",
            "Understanding your data early prevents many common ML mistakes."
        )

    return (
        "Welcome to ModelCraft.",
        "Start by uploading a dataset.",
        "Most ML projects begin with understanding the data."
    )


def _build_context_summary() -> str:

    parts = []

    df: pd.DataFrame | None = st.session_state.get("dataset")

    if df is not None:
        parts.append(
            f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns."
        )

        parts.append(
            f"Columns: {', '.join(df.columns[:10])}"
            + (" ..." if len(df.columns) > 10 else "")
        )

    target = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type")

    if target:
        parts.append(f"Target column: {target}")

    if task_type:
        parts.append(f"Problem type: {task_type}")

    if "preprocessor" in st.session_state:
        parts.append("Preprocessing pipeline has been configured.")

    model_name = st.session_state.get("trained_model_name")

    if model_name:
        parts.append(f"Trained model: {model_name}")

    if not parts:
        return "No dataset or model has been created yet."

    return "\n".join(parts)


def _call_llm(prompt: str):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "No AI backend configured. "
            "Please add GROQ_API_KEY."
        )

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

    if "advisor_messages" not in st.session_state:
        st.session_state["advisor_messages"] = []

    st.markdown("## 🧠 AI Mentor")

    st.markdown(
        f"""
**{observation}**

{recommendation}
"""
    )

    st.caption(f"💡 {note}")

    st.markdown("---")

    # =====================================================
    # CONVERSATION
    # =====================================================

    st.markdown("### Chat")

    with st.container(
        height=250,
        border=True
    ):

        if not st.session_state["advisor_messages"]:

            st.caption(
                "Ask a question to start a conversation."
            )

        else:

            for msg in st.session_state["advisor_messages"]:

                role = (
                    "user"
                    if msg["role"] == "user"
                    else "assistant"
                )

                with st.chat_message(role):
                    st.write(msg["content"])

    st.markdown("### Ask a Question")

    question = st.text_area(
        "",
        placeholder="Why is my accuracy low?",
        label_visibility="collapsed",
        height=80,
    )

    col1, col2 = st.columns(2)

    with col1:

        send = st.button(
            "Send",
            use_container_width=True,
        )

    with col2:

        if st.button(
            "Clear",
            use_container_width=True,
        ):
            st.session_state["advisor_messages"] = []
            st.rerun()

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    if send and question.strip():

        st.session_state["advisor_messages"].append(
            {
                "role": "user",
                "content": question.strip()
            }
        )

        context = _build_context_summary()

        prompt = textwrap.dedent(
            f"""
            You are an ML mentor helping beginners use ModelCraft.

            Current project context:

            {context}

            User question:

            {question}

            Rules:
            - Use simple language
            - Avoid jargon
            - Give practical next steps
            - Keep answers under 150 words
            """
        ).strip()

        with st.spinner("Thinking..."):
            answer = _call_llm(prompt)

        st.session_state["advisor_messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()