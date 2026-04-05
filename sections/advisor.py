import os
import textwrap
import streamlit as st

import pandas as pd


def _build_context_summary() -> str:
    """Summarize current state (dataset, target, task, model) for the LLM."""
    parts = []

    df: pd.DataFrame | None = st.session_state.get("dataset")
    if df is not None:
        parts.append(
            f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns. "
            f"Columns: {list(df.columns)[:10]}{' ...' if df.shape[1] > 10 else ''}."
        )

    target = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type")
    if target:
        parts.append(f"Target column: {target}.")
    if task_type:
        parts.append(f"Problem type: {task_type}.")

    if "preprocessor" in st.session_state:
        parts.append("Preprocessing: a ColumnTransformer pipeline is configured.")

    model_name = st.session_state.get("trained_model_name")
    if model_name:
        parts.append(f"Model: {model_name} has been trained.")

    if not parts:
        return "No dataset or model is loaded yet."

    return " ".join(parts)


def _call_llm(prompt: str) -> str:
    """
    Call a backend LLM.

    For now this is a simple placeholder. You can:
    - Plug in a free hosted API (e.g. Groq) using an API key in an env var.
    - Or use a local server (e.g. ollama / llama.cpp) and POST to http://localhost.

    Replace this body when you pick a backend.
    """

    # Example sketch for a Groq-like API (commented out so code runs without it):
    #
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "I don't have access to an LLM API yet. "
            "Set GROQ_API_KEY in your environment to enable live answers."
        )
    
    import requests
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

    # Placeholder answer so UI works even without a backend:
    return (
        "I am a placeholder advisor. Right now I don't call a real LLM, "
        "but I can still remind you of the steps:\n\n"
        "- Upload a dataset and pick the target.\n"
        "- Run EDA to understand distributions and missing values.\n"
        "- Configure preprocessing and split train/test.\n"
        "- Train one or more models and compare their performance."
    )


def advisor_panel():
    """Right-hand side advisor panel with chat-like interaction."""
    if "advisor_messages" not in st.session_state:
        st.session_state["advisor_messages"] = []

    st.markdown("### Ask ModelCraft")

    st.caption(
        "Ask questions in simple language. The assistant will use the current "
        "dataset, target, and model choices as context."
    )

    # Show history
    for msg in st.session_state["advisor_messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Advisor:** {content}")

    # Input box
    user_input = st.text_area(
        "Type your question here:",
        key="advisor_input",
        height=80,
        placeholder="E.g. Which model should I try next? Why is my accuracy low?",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ask = st.button("Ask advisor")
    with col2:
        clear = st.button("Clear chat")

    if clear:
        st.session_state["advisor_messages"] = []
        # st.experimental_rerun()

    if ask and user_input.strip():
        # Append user message
        st.session_state["advisor_messages"].append(
            {"role": "user", "content": user_input.strip()}
        )

        # Build context and call backend (or placeholder)
        context = _build_context_summary()
        full_prompt = textwrap.dedent(
            f"""
            You are a friendly ML tutor helping a beginner use a Streamlit app called ModelCraft.

            Current context:
            {context}

            User question:
            {user_input.strip()}

            Answer in simple, non-technical language and give concrete next steps.
            """
        ).strip()

        with st.spinner("Advisor is thinking..."):
            answer = _call_llm(full_prompt)

        st.session_state["advisor_messages"].append(
            {"role": "assistant", "content": answer}
        )

        # Clear input box
        # st.session_state["advisor_input"] = ""
        # st.experimental_rerun()