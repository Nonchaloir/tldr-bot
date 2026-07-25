"""
Document & Image Summarizer — FastAPI Backend
Step 8: SQLite chat history.

What changed:
  - sqlite3 database added to persist chat history across sessions
  - init_db() creates the messages table on startup if it doesn't exist
  - GET /history — returns all past messages from SQLite
  - DELETE /history — clears all messages from SQLite
  - /chat and /chat/stream now write to SQLite after each exchange
  - InMemoryChatMessageHistory is now seeded from SQLite on startup
    so in-session memory and on-disk history stay in sync

Run with: uvicorn api:app --reload
"""

import io
import gc
import os
import sqlite3
from threading import Thread

import easyocr
import numpy as np
import pdfplumber
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# LangChain imports
# RecursiveCharacterTextSplitter — smarter than our manual chunk_text():
# it tries to split on paragraphs first, then sentences, then words, then characters
# so chunks don't cut in the middle of a sentence

## ["\n\n", "\n", " ", ""] -> RecursiveCharacterTextSplitter  hierachy of separator 
## Smarter way of splitting long texts, espically for long lectures/long files that contains long text files
## If we use normal splitter -> cut hard at arbitrary token limits, which would slice sentences into half. 
## To solve this, they cut at every prargraph -> every sentence, to keep meaning intact for the chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document is LangChain's wrapper for a piece of text + metadata
# We convert our extracted text into Document objects so LangChain can work with them

## Unstructured text -> uniform data structure
## This is a wrapper to allow text to be readable by LangChain -> by wrapping our text + metadata
from langchain_core.documents import Document

# Chroma — LangChain's wrapper around ChromaDB
# Handles embedding + storing + querying in one object instead of three separate steps

## High-level wrapper for Chroma vector database
## Usually we would call embedding model -> format vector (embede them) -> store them into Chroma DB)
## But with Chroma wrapper, it combaines (embedding generation + index storage + search) into just 1 class
from langchain_chroma import Chroma

# HuggingFaceEmbeddings — LangChain's wrapper around SentenceTransformer
# Lets us pass our embedding model into LangChain components directly

## Connects local embedding models from HuggingFace to LangChain
## Helps us convert raw text strings into vecetor representation for our machine to understand
from langchain_huggingface import HuggingFaceEmbeddings

# InMemoryChatMessageHistory — stores the last N messages from the current session
# Seeded from SQLite on startup so memory is restored after a restart

## Tracks and retrain message history in memory for duration of an ACTIVE SESSION
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Load environment variables from .env file
# --------------------------------------------------------------------------

## Load our env file and load our Token as well as the model we are using into our api
load_dotenv()
HF_TOKEN     = os.getenv("HF_TOKEN")
BASE_MODEL   = os.getenv("BASE_MODEL")
ADAPTER_REPO = os.getenv("ADAPTER_REPO")

## Token is like the number of chunk of text, so for chat, it is much lesser so it will reply lesser as compared to the summary which has more
## Like normal chatgpt/claude, the token is how much they will think and then reply based on the max token we set
## Since it is our own model, we can set as many token as we want but more dosent mean good
MAX_CHAT_TOKENS = 350

## When doing summary, our model has to do more work and reply more, hence we set more Tokens for it
## The model stops generating words or stops after it hits the limit of 1500 tokens
MAX_SUMMARY_TOKENS = 1500

## The model can only read up to about 30k tokens, hence if the document is more, we have to ensure that it falls within the token limit or else it will crash
## Variable is set so that we know the limit and ensure that it does not go above it
MAX_INPUT_TOKENS = 30000

# Path to the SQLite database file — created automatically on first run
# Sits next to api.py so it's easy to find and back up

## Our database path for easy access
DB_PATH = "./chat_history.db"

## Creating the server application by doing FastAPI(), so you can just do @app in the future
app = FastAPI(title="Doc Summarizer API")

# --------------------------------------------------------------------------
# CORS
## middleware is code that sits between the incoming request and the endpoint function
## CORSMiddleware handles the preflight approval from the browser -> it checks the allow_origins list
## allow_methods and allow_headers with asterisk mean just allow everything
# --------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

## Prompt the AI so that it knows what to do whenever we call it, this is to also make sure we tackle problems that our model have such as hallucination
SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions based on the provided context.

If asked about a specific book, film, TV show, person, statistic, survey, or event that you cannot verify exists in your knowledge, say clearly that you cannot find reliable information about it rather than generating a plausible-sounding description. Never fabricate plot details, biographies, statistics, or historical specifics you cannot verify.

Always base your answers on the context provided to you."""

# --------------------------------------------------------------------------
# Summary prompts — one per length setting
# --------------------------------------------------------------------------

## Based on my suggestion just now
## Have different level of prompts so that we get different level of response, currently all set to Medium
## Hence we will get a summarised detail of the document of like the main point + explanation of the main point, with a tldr at the bottom
SUMMARY_PROMPTS = {
    "Short": """Analyse this document and produce a concise breakdown:

For each of the 2-3 most important concepts in the document, write:
**[Concept Name]**
→ What it is: one clear sentence definition
→ Why it matters: one sentence

Then end with:
**TL;DR:** One paragraph summarising the whole document in simple language.

Document:
{text}""",

    "Medium": """You are creating revision notes.

FIRST:
List every heading found in the document and write 5-10 bullet points explaining it.

Document:
{text}""",

    "Detailed": """Analyse this document thoroughly and produce an in-depth study breakdown:

Identify all major concepts or topics covered. For each one, write:

**[Number]. [Concept Name]**
→ **What it is:** A thorough explanation in 3-5 sentences covering what it means and how it works.
→ **Key details:** Any important conditions, formulas, rules, or nuances mentioned in the document.
→ **Example:** A worked example or concrete illustration that shows the concept in action.

After covering all concepts, end with:

**TL;DR**
Write 4-6 sentences summarising the entire document in plain, simple language — as if explaining to a friend who knows nothing about the topic.

**Likely Exam Topics**
Based on this document, list 3-5 specific things that are most likely to be tested.

Document:
{text}""",
}

# --------------------------------------------------------------------------
# SQLite setup
# --------------------------------------------------------------------------

def init_db():
    """
    Creates the messages table if it doesn't exist yet.
    Called once at startup — safe to call every time because of IF NOT EXISTS.

    Table columns:
      id        — auto-incrementing integer, unique row identifier
      role      — "user" or "assistant"
      content   — the message text
      timestamp — automatically set to the current time when the row is inserted
    """
    # connect() opens the database file, or creates it if it doesn't exist
    # The with block auto-commits on success and auto-rolls back on error
    with sqlite3.connect(DB_PATH) as conn:

        ## CREATE TABLE IF NOT EXISTS -> safe to execute everytime
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def db_save_message(role: str, content: str):
    """
    Inserts one message row into the messages table.
    Called after every user question and every model answer.
    """
    # ? placeholders prevent SQL injection — values are passed separately
    with sqlite3.connect(DB_PATH) as conn:

        ## Insert message into db with values of the role -> user/assistant and content -> message/reply
        conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content),
        )

def db_load_messages() -> list[dict]:
    """
    Returns all messages from the database in chronological order.
    Each row comes back as {"role": "user"/"assistant", "content": "..."}.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # row_factory makes each row a dict instead of a plain tuple
        # so we can access columns by name: row["role"] instead of row[0]

        ## Converts query results from plain tuples -> dict like row objects
        ## Access column values by COLUMN NAME and by INDEX
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content FROM messages ORDER BY id ASC"
        ).fetchall()

        ## returns rows like {"role": _____________, "content": _____________________}
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def db_clear_messages():
    """Deletes all rows from the messages table."""
    with sqlite3.connect(DB_PATH) as conn:

        ## delete all messages by the user
        conn.execute("DELETE FROM messages")

# --------------------------------------------------------------------------
# Load models at startup
# --------------------------------------------------------------------------

# Create the SQLite table before anything else runs
# This must happen before the model loads so the DB is ready when endpoints are called

## Create and initilse DB
init_db()

# EasyOCR — placeholder until your Qwen VL model is ready
# gpu=False forces CPU to avoid sm_120 compatibility crash
ocr_reader = easyocr.Reader(["en"], gpu=False)

## For now, we will be using embedding_model from HuggingFace free one, we do not need to train our own
## This embedding model will take in words/sentence and convert them into numbers so we can store in ChromaDB
## We now wrap it in HuggingFaceEmbeddings so LangChain components can use it directly
## model_kwargs passes gpu device setting; encode_kwargs normalises the vectors to unit length
## (normalisation makes similarity search more accurate)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

## Login into our Model on Hugging Face
login(token=HF_TOKEN)

## Clear our VRAM so that we have enough space, this is probably only for our com setup with only 8gb to spare
torch.cuda.empty_cache()
gc.collect()

## Load the model in 4bit, so it fits nicely in the 16gb VRAM instead of loading it at 13gb
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    ## Store in 4-bit -> compute in bfloat16 (16-bit) -> discard
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

## Load from our base model, converts Text into tokens (tokens back into text)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 4-bit quantization
# device_map="auto" lets PyTorch figure out the best device — matches your eval script
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    ## Let PyTorch decide whether to use GPU or CPU on its own
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)

## Clear VRAM again
torch.cuda.empty_cache()
gc.collect()

# Stack the LoRA adapter on top of the base model
# When you finish a new phase, just update ADAPTER_REPO in .env — nothing else changes

## All the training we have done, the weights, we will apply onto the Base Model
llm = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
## Switch the model into evaluation mode which drops certain layers to prevent overfitting
llm.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Model ready: {ADAPTER_REPO}")

# --------------------------------------------------------------------------
# LangChain setup
# --------------------------------------------------------------------------

# text_splitter replaces our manual chunk_text() function
# RecursiveCharacterTextSplitter is smarter — it tries to split on "\n\n" first
# (paragraph breaks), then "\n" (line breaks), then " " (words), then "" (characters)
# This means chunks almost never cut in the middle of a sentence
# chunk_size=500 and chunk_overlap=50 matches what our old chunk_text() did

## As stated above, it replaces the previous function to split chunks more effectively 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Chroma vector store — replaces our manual chroma_client + collection setup
# persist_directory="./chroma_db" means it saves to disk just like before
# embedding_function tells it which model to use when converting text to numbers
# collection_name="documents" keeps it in the same collection as before

## ChromaDB sits on our current path, like a normal DB
## if it does not exist, chroma_db will be created, if not we will use what already existed

## As stated above, it replaces the previous procedure to store data into ChromaDB
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="documents",
)

## Container object that holds a list of chat messages, which is assigned to variable memory
## Stores the messages in computer's RAM so it remembers conversations
## Seeded from SQLite on startup so the model has full history context immediately
memory = InMemoryChatMessageHistory()

# Restore previous messages from SQLite into in-session memory
# so the model has full conversation context from the moment the server starts
# Without this, memory would be empty on every restart even though SQLite has the history

## Load previous messages
## If exist, load them so the user can see past history conversation
for msg in db_load_messages():
    if msg["role"] == "user":
        memory.add_message(HumanMessage(content=msg["content"]))
    else:
        memory.add_message(AIMessage(content=msg["content"]))

# --------------------------------------------------------------------------
# Request / Response models (Pydantic)
# --------------------------------------------------------------------------

## Streamlit will send data in a form of text + length, will be used to validate the data type of the messages being sent to our backend
## Over at app.py, we will send in the following, {"text": _____, "length": _____}
class SummarizeRequest(BaseModel):
    text: str
    length: str   # "Short", "Medium", or "Detailed"

## This is what we will respond Streamlit, our app.py with. It will receive {"summary": _____}
class SummarizeResponse(BaseModel):
    summary: str

## Streamlit (app.py) will send data in the form of a dictionary question
## json={"question": question}
class ChatRequest(BaseModel):
    question: str

## This is what our API will return back to Streamlit (app.py)
## response.json()["answer"]
class ChatResponse(BaseModel):
    answer: str

## New class for our ChromaDB
## Checks the validity of the data being sent FROM Streamlit TO the backend
## Accepts data in the form of {"filename":_____, "text":______}
class StoreRequest(BaseModel):
    filename: str
    text: str

## New class for our ChromaDB
## Will send back in the form of the dictionary {"chunks_stored": _____} to our frontend
class StoreResponse(BaseModel):
    chunks_stored: int

# --------------------------------------------------------------------------
# Long input handling — unchanged
# --------------------------------------------------------------------------

## This is for convo with the LLM to ask question about the file
def truncate_text(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """
    Tokenizes the text and truncates it to max_tokens if it's too long.
    Returns the truncated text as a string, not as token IDs.

    Why tokenize instead of just counting characters?
    Because 1 token ≠ 1 character — some words are 1 token, some are 3 or 4.
    Counting characters would give an inaccurate estimate.
    Using the actual tokenizer gives an exact count.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        ## If the length of the text is less than the max token, just return the text
        return text

    truncated_ids = token_ids[:max_tokens]

    ## Else, we need to truncate the text because its too long
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    print(f"[truncate_text] Input was {len(token_ids)} tokens — truncated to {max_tokens}")
    return truncated_text

# --------------------------------------------------------------------------
# generate() — non-streaming, used by /summarize
# Unchanged — LangChain doesn't help with streaming summarization
# --------------------------------------------------------------------------
def generate(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS) -> str:
    # apply_chat_template formats the message list into the exact prompt string
    # the model was trained on — adds the right special tokens for Qwen

    ## For the model, it has to be in a specific format to be able to read the text
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    ## SEnds the input to the same device as where our LLM sits
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    ## full gets both the input + the model answer
    full = tokenizer.decode(output[0], skip_special_tokens=True)

    ## get the length of our input so we can string slice it away 
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full[len(prompt_text):].strip()

# --------------------------------------------------------------------------
# generate_stream() — streaming, used by /summarize/stream
# Unchanged — LangChain doesn't help with streaming summarization
# --------------------------------------------------------------------------
def generate_stream(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    ## skip_prompt=True — don't stream the prompt back, only new tokens
    ## skip_special_tokens=True — don't stream <|im_end|> etc.
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    ## model.generate() runs in a background thread so it doesn't block
    ## the generator from yielding tokens as they arrive
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = Thread(target=llm.generate, kwargs=generation_kwargs)
    thread.start()

    ## yield each token as it arrives from the streamer queue
    for token in streamer:
        yield token

    thread.join()

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

## Check the status of the API
@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# NEW: GET /history
# Returns all past messages from SQLite so Streamlit can restore the chat
# UI on startup without the user needing to re-ask everything
# --------------------------------------------------------------------------
@app.get("/history")
async def get_history():
    # db_load_messages() runs SELECT role, content FROM messages ORDER BY id ASC
    # returns a list of dicts: [{"role": "user", "content": "..."}, ...]

    ## Get all the past conversations from the db 
    messages = db_load_messages()
    return {"messages": messages}


# --------------------------------------------------------------------------
# NEW: DELETE /history
# Wired to the "Clear all" button in Streamlit
# Clears both SQLite (on disk) and InMemoryChatMessageHistory (in RAM)
# so history is gone from everywhere, not just the UI
# --------------------------------------------------------------------------
@app.delete("/history")
async def clear_history():
    # DELETE FROM messages — wipes all rows from the table

    ## Delete ALL MESSAGES from the table
    db_clear_messages()
    # clear() resets the in-session memory list to empty
    memory.clear()
    return {"status": "cleared"}


## Extract detail from the file we upload — unchanged
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    filename = file.filename.lower()

    # await pauses here until the file bytes are fully received,
    # but lets FastAPI handle other requests in the meantime
    contents = await file.read()

    # ── PDF ──────────────────────────────────────────────────────────────
    if filename.endswith(".pdf"):
        try:
            ## io.BytesIO converts raw bytes into a file-like object so the types match what pdfplumber expects
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            return {"text": "\n\n".join(pages_text)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")

    # ── Image ─────────────────────────────────────────────────────────────
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        try:
            ## Same type mismatch fix — raw bytes → file-like → PIL Image
            image = Image.open(io.BytesIO(contents))
            image_np = np.array(image)
            ## We use the ocr_reader to read the data
            results = ocr_reader.readtext(image_np)
            extracted = " ".join([block[1] for block in results])
            return {"text": extracted}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image OCR failed: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")


# --------------------------------------------------------------------------
# POST /store — LangChain chunking + ChromaDB, unchanged from Step 7
# --------------------------------------------------------------------------
@app.post("/store", response_model=StoreResponse)
async def store(request: StoreRequest):
    ## Check ChromaDB first — if this file is already stored, skip it
    ## This prevents duplicate chunks when the app restarts and re-uploads the same file
    existing = vector_store.get(where={"filename": request.filename})
    if existing and existing.get("ids"):
        print(f"[store] {request.filename} already in ChromaDB, skipping")

        ## already exist dont need to store again
        return StoreResponse(chunks_stored=0)

    # text_splitter.create_documents() splits the text AND wraps each chunk
    # in a LangChain Document object with the filename stored as metadata
    # This replaces our manual chunk_text() + manual metadata dict

    ## as stated for text_splitter, splits the file details up into hierachy
    ## we will use the filename as metadata
    docs = text_splitter.create_documents(
        texts=[request.text],
        metadatas=[{"filename": request.filename}],
    )

    ## vector_store.add_documents() embeds each chunk and stores it in ChromaDB
    ## This replaces embedding_model.encode() + collection.add() — one line instead of three
    vector_store.add_documents(docs)

    ## We then return to the frontend the number of chunks we stored
    return StoreResponse(chunks_stored=len(docs))


# --------------------------------------------------------------------------
# POST /summarize — non-streaming fallback, unchanged
# --------------------------------------------------------------------------
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])

    # truncate the document text if it's too long for the model's context window
    safe_text = truncate_text(request.text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_template.format(text=safe_text)},
    ]

    summary = generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS)
    return SummarizeResponse(summary=summary)


# --------------------------------------------------------------------------
# POST /summarize/stream — unchanged
# --------------------------------------------------------------------------
@app.post("/summarize/stream")
async def summarize_stream(request: SummarizeRequest):
    prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])

    # truncate the document text if it's too long for the model's context window
    safe_text = truncate_text(request.text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_template.format(text=safe_text)},
    ]

    return StreamingResponse(
        generate_stream(messages, max_new_tokens=MAX_SUMMARY_TOKENS),
        media_type="text/plain",
    )


# --------------------------------------------------------------------------
# POST /chat — non-streaming fallback
# Now also writes to SQLite after generating the answer
# --------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    ## Vector database is being converted into a retrieval interface
    ## When searched, return TOP 3 MOST SIMILAR DOCUMENT CHUNKS
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    ## Query to the databsae, which embeds the user's input into a vector -> queries the database -> returns a list containing 3 matching Document Objects
    relevant_docs = retriever.invoke(request.question)

    ## Makes it into a proper text that is readable 
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    ## Converts raw message objects stored in the "memory" into readable text diaologue transcript
    history = "\n".join([
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in memory.messages
    ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Previous conversation:\n{history}\n\n" if history else ""
            f"Answer the question using only the context below. "
            f"If the context doesn't contain enough information to answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.question}"
        )},
    ]

    ## Retrieve answer from our LLM
    answer = generate(messages, max_new_tokens=MAX_CHAT_TOKENS)

    ## Storing user's prompt by wrapping the raw query string into a HumanMessage object -> add into memory store
    memory.add_message(HumanMessage(content=request.question))

    ## Storing Model's answer by wrapping the generated string into an AIMessage object -> add into memory store
    memory.add_message(AIMessage(content=answer))

    # Write both messages to SQLite so they survive a server restart

    ## Now we also store the messages into our database (db)
    ## Store BOTH user + assistant answer in
    db_save_message("user", request.question)
    db_save_message("assistant", answer)

    return ChatResponse(answer=answer)


# --------------------------------------------------------------------------
# POST /chat/stream — streaming version
# Also writes to SQLite inside stream_and_save() after all tokens arrive
# --------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    ## Same explanation as chat
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(request.question)

    ## Join the retrieved document chunks into one context string
    ## relevant_docs is a list of LangChain Document objects — .page_content is the text
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    ## Get chat history from memory so the model has context from earlier in the session
    history = "\n".join([
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in memory.messages
    ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Previous conversation:\n{history}\n\n" if history else ""
            f"Answer the question using only the context below. "
            f"If the context doesn't contain enough information to answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.question}"
        )},
    ]

    def stream_and_save():
        ## When we retrieve the answer from the model, we will get the answer by yield token
        full_answer = ""
        for token in generate_stream(messages, max_new_tokens=MAX_CHAT_TOKENS):
            full_answer += token
            yield token

        ## same as chat explanation — save both sides of the exchange to memory and SQLite
        memory.add_message(HumanMessage(content=request.question))
        memory.add_message(AIMessage(content=full_answer))

        # Write to SQLite after streaming finishes so the exchange is persisted to disk

        ## Same as chat function, we save the user + assistant answer to our db
        db_save_message("user", request.question)
        db_save_message("assistant", full_answer)

    return StreamingResponse(stream_and_save(), media_type="text/plain")