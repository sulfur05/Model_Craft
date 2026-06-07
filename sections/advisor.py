import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _mentor_content():

    if "trained_model" in st.session_state:
        return (
            "🎉 Your model finished training.",
            "I'd explore Explainability next.",
            "Feature importance often reveals surprising insights."
        )

    if "preprocessor" in st.session_state:
        return (
            "✅ Your data is ready.",
            "I'd train a model next.",
            "Start with a baseline before tuning."
        )

    if st.session_state.get("eda_complete", False):
        return (
            "📊 Nice work exploring the data.",
            "I'd configure preprocessing next.",
            "Good preprocessing often matters more than model choice."
        )

    if "dataset" in st.session_state:
        return (
            "📁 Your dataset is uploaded.",
            "I'd run EDA next.",
            "Understanding the data early prevents many ML mistakes."
        )

    return (
        "👋 Welcome to ModelCraft.",
        "Start by uploading a dataset.",
        "Every ML workflow begins with understanding the data."
    )


def _call_llm(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Missing GROQ_API_KEY."

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

    st.markdown(
        """
        <div class="mentor-title">
            🧠 AI Mentor
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mentor-card">

        <div>
        {observation}
        </div>

        <div class="mentor-recommendation">
        {recommendation}
        </div>

        <div style="
            font-size:0.9rem;
            opacity:0.8;
            margin-top:10px;
        ">
        💡 {note}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.popover("💬 Ask Mentor"):

        question = st.text_area(
            "Question",
            placeholder="Why is my accuracy low?",
            height=100,
        )

        if st.button(
            "Send",
            use_container_width=True,
            key="mentor_send"
        ):

            if not question.strip():
                return

            prompt = f"""
You are ModelCraft AI Mentor.

The user is a beginner learning machine learning.

Question:
{question}

Rules:
- Maximum 80 words
- Use simple language
- Give practical advice
- Avoid jargon
"""

            with st.spinner("Thinking..."):
                answer = _call_llm(prompt)

            st.success(answer)