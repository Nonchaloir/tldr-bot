"""
Document & Image Summarizer — Streamlit UI
Timeout fix: explicit (connect, read) timeouts added to streaming requests
so long map-reduce summarization on the 9B model doesn't get killed early
by requests' default timeout behavior.

Run order:
  Terminal 1: uvicorn api:app --reload     (local dev)
  Terminal 2: streamlit run Home.py
  OR: set API_URL to your Modal endpoint and just run Streamlit
"""

import os
import uuid
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="TLDR BOT — Chat",
    page_icon="💬",
    layout="wide",
)

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "md", "txt"]

# In Docker/Streamlit Cloud, API_URL is set via environment variable / secrets
# Locally it falls back to localhost:8000 so nothing breaks when running without it
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "summaries" not in st.session_state:
    st.session_state.summaries = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    try:
        response = requests.get(
            f"{API_URL}/history",
            params={"session_id": st.session_state.session_id},
            timeout=(10, 30),  # (connect, read) — history load should be near-instant
        )
        st.session_state.chat_history = response.json().get("messages", [])
    except Exception:
        st.session_state.chat_history = []

# --------------------------------------------------------------------------
# Functions that talk to FastAPI / Modal
# --------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    response = requests.post(
        f"{API_URL}/extract",
        files={"file": (uploaded_file.name, uploaded_file.read(), "application/octet-stream")},
        headers={"X-Requested-With": "XMLHttpRequest"},
        # PDF extraction of a large file can take a little while — generous read timeout
        timeout=(30, 300),
    )
    return response.json()["text"]


def store_text(filename: str, text: str):
    response = requests.post(
        f"{API_URL}/store",
        json={"filename": filename, "text": text, "session_id": st.session_state.session_id},
        timeout=(30, 120),
    )
    return response.json()["chunks_stored"]


def summarize_text(text: str, length: str):
    # NEW: explicit (connect_timeout, read_timeout) tuple.
    # connect_timeout=30 — fail fast if Modal is unreachable.
    # read_timeout=900 — matches modal_app.py's server-side timeout=900.
    # Without this, requests' default read behavior can drop a long streaming
    # response well before the server actually finishes, especially during
    # the quiet gaps between chunk summaries in the long-document map-reduce path.
    with requests.post(
        f"{API_URL}/summarize/stream",
        json={"text": text, "length": length},
        stream=True,
        timeout=(30, 900),
    ) as response:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def answer_question(question: str):
    # Chat responses are much shorter than summaries, so a shorter read timeout is fine
    with requests.post(
        f"{API_URL}/chat/stream",
        json={"question": question, "session_id": st.session_state.session_id},
        stream=True,
        timeout=(30, 300),
    ) as response:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


# --------------------------------------------------------------------------
# Sidebar — settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    summary_length = st.select_slider(
        "Summary length",
        options=["Short", "Medium", "Detailed"],
        value="Medium",
    )
    st.caption(f"Supported files: {', '.join(SUPPORTED_TYPES)}")
    st.divider()

    if st.button("🗑️ Clear all", use_container_width=True):
        st.session_state.summaries = {}
        st.session_state.chat_history = []
        try:
            requests.delete(
                f"{API_URL}/history",
                params={"session_id": st.session_state.session_id},
                timeout=(10, 30),
            )
        except Exception:
            pass
        st.rerun()


# --------------------------------------------------------------------------
# Main area — upload + summarize
# --------------------------------------------------------------------------
st.title("📄 Document & Image Summarizer")
st.write("Upload one or more documents or images. I'll pull out the key details.")

uploaded_files = st.file_uploader(
    "Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Uploaded files")
    for f in uploaded_files:
        st.write(f"📎 **{f.name}** — {f.size / 1024:.1f} KB")

    if st.button("✨ Summarize all", type="primary"):
        progress = st.progress(0, text="Starting...")
        for i, f in enumerate(uploaded_files):
            if f.name not in st.session_state.summaries:
                with st.spinner(f"Reading {f.name}..."):
                    text = extract_text(f)

                with st.spinner(f"Storing {f.name} in memory..."):
                    chunks_stored = store_text(f.name, text)
                    st.caption(f"Stored {chunks_stored} chunks from {f.name}")

                with st.expander(f"📄 {f.name} — generating...", expanded=True):
                    summary = st.write_stream(summarize_text(text, summary_length))
                st.session_state.summaries[f.name] = summary

            progress.progress((i + 1) / len(uploaded_files), text=f"Done with {f.name}")
        progress.empty()
        st.success("All files summarized!")

# --------------------------------------------------------------------------
# Show summaries
# --------------------------------------------------------------------------
if st.session_state.summaries:
    st.subheader("📝 Key details")
    for filename, summary in st.session_state.summaries.items():
        with st.expander(f"📄 {filename}", expanded=True):
            st.markdown(summary)

st.divider()

# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
st.subheader("💬 Ask about your documents")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about the uploaded files..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(answer_question(question))

    st.session_state.chat_history.append({"role": "assistant", "content": answer})