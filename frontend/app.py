import uuid

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# `thread_id` is the conversation ID sent to the backend on every /chat call
# (same concept as thread_id in your original script — same ID means the
# agent remembers prior turns; a new one means a clean slate).
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content) tuples, for display only

# ---------------------------------------------------------------------------
# Sidebar: upload PDFs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Upload documents")

    uploaded_files = st.file_uploader(
        "Add PDFs to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded files") and uploaded_files:
        files_payload = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in uploaded_files
        ]
        with st.spinner("Embedding and indexing..."):
            resp = requests.post(f"{BACKEND_URL}/upload", files=files_payload)

        if resp.ok:
            data = resp.json()
            st.success(f"Indexed {data['chunks_added']} chunks from {data['uploaded']}")
        else:
            st.error(f"Upload failed: {resp.text}")

    st.divider()

    if st.button("Start new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Thread ID: `{st.session_state.thread_id}`")

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------

st.title("Chat with your documents")

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

user_input = st.chat_input("Ask something about your documents...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": user_input, "thread_id": st.session_state.thread_id},
                    timeout=60,
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend. Is it running on http://localhost:8000?")
                st.stop()

        if resp.ok:
            data = resp.json()
            answer = data["answer"]
            source = data.get("source")
            st.write(answer)
            if source:
                st.caption(f"Source: {source}")
            st.session_state.chat_history.append(("assistant", answer))
        else:
            error_msg = f"Error: {resp.text}"
            st.error(error_msg)
            st.session_state.chat_history.append(("assistant", error_msg))
