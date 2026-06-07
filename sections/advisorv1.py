import os
import textwrap
import streamlit as st

import pandas as pd
from dotenv import load_dotenv
load_dotenv()



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
            "model": "llama-3.3-70b-versatile",
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

    # 🔴 CHANGE 6: Proactive advisor tips
    # Add automatic tips at key milestones
    if "dataset" in st.session_state and len(st.session_state["advisor_messages"]) == 0:
        dataset = st.session_state["dataset"]
        st.session_state["advisor_messages"].append({
            "role": "advisor",
            "content": f"Great! I see you have {dataset.shape[0]:,} rows and {dataset.shape[1]} features. "
                      f"Next, let's explore your data in the EDA section to understand patterns and distributions."
        })
    
    if "preprocessor" in st.session_state and "advisor_tip_preprocessing" not in st.session_state:
        st.session_state["advisor_messages"].append({
            "role": "advisor",
            "content": "✅ Data is ready! Your preprocessing pipeline is configured. "
                      "Now let's train a model. Random Forest is a great starting point for most datasets!"
        })
        st.session_state["advisor_tip_preprocessing"] = True
    
    if "trained_model" in st.session_state and "advisor_tip_training" not in st.session_state:
        accuracy = st.session_state.get("accuracy", st.session_state.get("r2_score", 0))
        metric_name = "accuracy" if "accuracy" in st.session_state else "R² score"
        st.session_state["advisor_messages"].append({
            "role": "advisor",
            "content": f"🎯 Model trained! Your {metric_name}: {accuracy:.2%}. "
                      f"Next, let's explore the Explainability section to understand which features drove predictions!"
        })
        st.session_state["advisor_tip_training"] = True

    st.markdown("### Ask ModelCraft")

    st.caption(
        "Ask questions in simple language. The assistant will use the current "
        "dataset, target, and model choices as context."
    )

    # Show history in a scrollable container
    with st.container(border=True, height=400):
        if not st.session_state["advisor_messages"]:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #888;">
                    <p style="text-align: center; font-size: 14px;">
                        💬 Your chat will appear here<br>
                        <span style="font-size: 12px;">Ask a question to get started!</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["advisor_messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    st.markdown(f"**Advisor:** {content}")
            
            # Auto-scroll anchor
            st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <script>
                    let element = document.getElementById("chat-bottom");
                    if (element) {
                        element.scrollIntoView({behavior: "smooth", block: "end"});
                    }
                </script>
                """,
                unsafe_allow_html=True,
            )

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

        # Rerun to display the response immediately
        st.rerun()