"""
Document & Image Summarizer — Streamlit UI
Step 8: SQLite chat history.

What changed:
  - On startup, calls GET /history to restore past chat from SQLite
  - "Clear all" button now also calls DELETE /history to wipe SQLite
  - Everything else unchanged

Run order:
  Terminal 1: uvicorn api:app --reload
  Terminal 2: streamlit run Home.py
"""

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

# Page name, at the top
st.set_page_config(
    page_title="TLDR BOT — Chat",
    page_icon="💬",
    layout="wide",
)

# Supported type of document person can upload
SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg"]

# The address FastAPI is running on.
# All requests from Streamlit go to this base URL.
# Defined once here so if the port ever changes, we only update it in one place.
import os
# In Docker, API_URL is set to "http://api:8000" via environment variable in docker-compose.yml
# Locally it falls back to localhost:8000 so nothing breaks when running without Docker
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --------------------------------------------------------------------------
# Session state — Streamlit reruns the whole script on every click, so
# anything you want to persist across reruns has to live here.
# --------------------------------------------------------------------------

## Keeps track of all the data/info of the chat
## summaries -> what we upload to the bot to summarise
## chat_history -> what WE message the bot, in the ask about your document, and what it replies
if "summaries" not in st.session_state:
    st.session_state.summaries = {}        # filename -> summary text

if "chat_history" not in st.session_state:
    # NEW: restore chat history from SQLite on first load
    # This runs once per browser session — if chat_history isn't in session_state
    # yet, it means the page just loaded, so we fetch past messages from the API
    # and pre-populate the list so the user sees their previous conversation immediately

    ## Page just loaded, lets try to retrieve all the old history from our db 
    try:

        ## Call to history to retrieve all the old conversation from db
        response = requests.get(f"{API_URL}/history")
        st.session_state.chat_history = response.json().get("messages", [])
    except Exception:
        # If the API isn't running yet, start with an empty history
        st.session_state.chat_history = []

# --------------------------------------------------------------------------
# Functions that talk to FastAPI
# --------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    # Send the file to FastAPI's /extract endpoint as a POST request.
    # "files" is how you send a file in a POST request — FastAPI receives
    # it as an UploadFile on the other end.
    # The tuple is: (filename, raw bytes, content type)

    ## Send over to our API side with the requests posting to our API_URL
    ## "application/octet-stream" is a content type, a label of what we are sending, the file
    ## "application/octet-stream"   → raw binary data (bytes), unknown format
    ## because we have many different type of file format, so we do not specify
    response = requests.post(
        f"{API_URL}/extract",
        files={"file": (uploaded_file.name, uploaded_file.read(), "application/octet-stream")},
    )
    # .json() parses the response FastAPI sends back into a Python dict
    # FastAPI returned {"text": "..."} so ["text"] grabs just the extracted text string

    ## we return the extracted text back to Streamlit, which saves it to session_state and displays it
    return response.json()["text"]


def store_text(filename: str, text: str):
    # NEW in Step 5: send the extracted text to /store so ChromaDB
    # chunks it, embeds it, and stores it for later retrieval during chat.
    # StoreRequest format: {"filename": ___, "text": ___}

    ## Sends it over to the API side, our backend, of the filename and the content that we extracted from the file
    ## So that we can use it for future reference etc or when we ask questions about the file,
    ## since the db stores the text split into many smaller chunks from the file
    ## API side of domain /store only accepts in the dictionary of {"filename": _____, "text": _____}
    response = requests.post(
        f"{API_URL}/store",
        json={"filename": filename, "text": text},
    )
    ## API side would then return the response and then we get the amount of chunks stored from the extracted text
    ## We then return it to let the user know for now how many chunks are stored
    return response.json()["chunks_stored"]


def summarize_text(text: str, length: str):
    # NEW: calls /summarize/stream instead of /summarize
    # Returns a generator that yields tokens one by one — same pattern as answer_question()
    # st.write_stream() in the UI will display each token as it arrives,
    # so the numbered breakdown appears section by section instead of all at once

    ## Same as answer_question(), stream=True keeps the connection open
    ## and reads tokens as they arrive instead of waiting for the full response
    with requests.post(
        f"{API_URL}/summarize/stream",
        json={"text": text, "length": length},
        stream=True,
    ) as response:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


## we can chat normally with the bot
def answer_question(question: str):
    # Calls /chat/stream — returns a generator that yields tokens one by one
    # st.write_stream() displays each token as it arrives word by word

    ## Same as above, we send a request to the API_URL but this time to the domain of chat
    ## stream=True keeps the connection open and reads tokens as they arrive
    ## instead of waiting for the whole response before returning
    with requests.post(
        f"{API_URL}/chat/stream",
        json={"question": question},
        stream=True,
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
    ## At the bottom of the slider, it will show these
    st.caption(f"Supported files: {', '.join(SUPPORTED_TYPES)}")
    ## Just a line to divide like a div
    st.divider()

    ## Button to clear all summaries and chat history we have
    if st.button("🗑️ Clear all", use_container_width=True):
        st.session_state.summaries = {}
        st.session_state.chat_history = []
        # NEW: also wipe SQLite and in-session memory on the API side
        # Without this, clearing the UI would have no effect on the database —
        # the history would come back the next time the page loads
        try:
            requests.delete(f"{API_URL}/history")
        except Exception:
            pass
        st.rerun()


# --------------------------------------------------------------------------
# Main area — upload + summarize
# --------------------------------------------------------------------------
st.title("📄 Document & Image Summarizer")
st.write("Upload one or more documents or images. I'll pull out the key details.")

## Something to do with the server side, when we upload a file, the server holds onto the files on their end
## So whenever the page refreshes/page load, the server just hands to st.file_uploader() the previous files that were already uploaded.
uploaded_files = st.file_uploader(
    "Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

## Then if there are any files that were uploaded
## We loop through the files and then show the name and the size of each file
## Then we have a button that helps to summarise all the files that we uploaded one by one
## Then the summarised text will be TRACKED/KEPT by using the dictionary (st.session_state.summaries)
if uploaded_files:
    st.subheader("Uploaded files")
    for f in uploaded_files:
        st.write(f"📎 **{f.name}** — {f.size / 1024:.1f} KB")

    if st.button("✨ Summarize all", type="primary"):
        progress = st.progress(0, text="Starting...")
        for i, f in enumerate(uploaded_files):
            # Only process files we haven't already summarized
            if f.name not in st.session_state.summaries:
                with st.spinner(f"Reading {f.name}..."):
                    ## Get the text from the file we uploaded, from bytes/Image -> text
                    text = extract_text(f)

                # store chunks in ChromaDB right after extracting
                # so the chat endpoint has something to search through
                with st.spinner(f"Storing {f.name} in memory..."):
                    ## store the text split into chunks in our DB alongside the filename
                    chunks_stored = store_text(f.name, text)
                    st.caption(f"Stored {chunks_stored} chunks from {f.name}")

                # NEW: stream the summary directly into a temporary expander
                # so the user sees the breakdown appearing section by section.
                # st.write_stream() returns the full completed text when done,
                # which we save to session_state so it persists across reruns.
                with st.expander(f"📄 {f.name} — generating...", expanded=True):
                    summary = st.write_stream(summarize_text(text, summary_length))
                st.session_state.summaries[f.name] = summary

            progress.progress((i + 1) / len(uploaded_files), text=f"Done with {f.name}")
        progress.empty()
        st.success("All files summarized!")

# --------------------------------------------------------------------------
# Show summaries — one expandable section per file
# --------------------------------------------------------------------------

## The summaries which our model previously summarised are being kept in st.session_state.summaries
## If there exist any summaries we had, it will loop through all and get the filename (key) and the summary (value)
## st.expander is just a collapsible section in the page, the content inside is the summary
## The with acts the same as for messages, it lets the system know which expander this is, which container it is, and then writes the summary to that container
## Then it renders the details in markdown -> plain text with formatting
if st.session_state.summaries:
    st.subheader("📝 Key details")
    for filename, summary in st.session_state.summaries.items():
        with st.expander(f"📄 {filename}", expanded=True):
            st.markdown(summary)

## Acts like a break/divider between sections
st.divider()

# --------------------------------------------------------------------------
# Chat — ask follow-up questions about the uploaded docs
# --------------------------------------------------------------------------

## Subheader to show words
st.subheader("💬 Ask about your documents")

## Whatever message in the chat_history gets printed again when the page reruns
## The with pattern allows st.chat_message to know WHICH container to write it on, user vs assistant
## Then once you tell st.chat_message() who the message belongs to (user vs assistant), it can write the message on the proper side
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

## := walrus operator
## Does 2 things: assigns the return value to question, and the whole expression is then checked
## question = st.chat_input("...") -> if question:
if question := st.chat_input("Ask a question about the uploaded files..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        ## st.write_stream() takes our generator from answer_question()
        ## displays each token as it arrives word by word in the chat bubble
        ## returns the full completed text at the end so we can save it to chat_history
        answer = st.write_stream(answer_question(question))

    st.session_state.chat_history.append({"role": "assistant", "content": answer})