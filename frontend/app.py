"""DataMind RAG Assistant - Streamlit chat frontend."""

from __future__ import annotations

import streamlit as st

from api_client import API_BASE_URL, APIClientError, ask_question, check_health

st.set_page_config(
    page_title="DataMind RAG Assistant",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 DataMind RAG Assistant")
    st.markdown(
        "A retrieval-augmented AI assistant for **Data Analysis, Python, "
        "SQL, and AI fundamentals** — answers are grounded in a document "
        "corpus, not the model's general knowledge."
    )

    st.divider()
    st.subheader("Backend status")
    health = check_health()
    if health is None:
        st.error(f"⚠️ Cannot reach backend at `{API_BASE_URL}`")
    elif not health.get("vector_store_loaded"):
        st.warning("⚠️ Backend is up, but the vector store is not loaded yet.")
    else:
        st.success(f"✅ Connected — {health.get('collection_count', '?')} chunks indexed")

    st.divider()
    st.caption("Ask me anything about:")
    st.caption("Python • Pandas • NumPy • SQL • EDA • Visualization • ML • AI")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("📊 DataMind RAG Assistant")
st.markdown("Ask me anything about **Data Analysis, AI, Python, SQL, and modern data tools.**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for source in msg["sources"]:
                    st.markdown(f"📄 {source}")

# Chat input
question = st.chat_input("e.g. What is the difference between Pandas and NumPy?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"📄 {source}")
                else:
                    st.caption("No sources were retrieved for this question.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except APIClientError as exc:
                error_message = f"⚠️ {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
