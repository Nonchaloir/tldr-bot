"""
TLDR BOT — Modal Deployment
Runs the FastAPI backend on Modal's cloud GPU infrastructure.

How it works:
  - Modal builds a container image with all dependencies
  - The container runs on an A10G GPU (24GB VRAM) on demand
  - Scales to zero when idle — you only pay when someone uses the app
  - Cold start (first request after idle): ~60-90 seconds to load model
  - Subsequent requests in same session: fast

Deploy with: modal deploy modal_app.py
Get endpoint URL from Modal dashboard after deploying

Local dev still uses: uvicorn api:app --reload
Modal is production only
"""

import io
import gc
import os
import sqlite3
from threading import Thread

import modal

# --------------------------------------------------------------------------
# Modal image — defines what gets installed in the container
# Start from a CUDA-enabled base so PyTorch can use the A10G GPU
# --------------------------------------------------------------------------

# Define the container image
# modal.Image.from_registry pulls a pre-built Docker image as the base
# We use nvidia/cuda so CUDA libraries are already present for PyTorch
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    # Install PyTorch nightly with cu128 support for CUDA 12.8
    # This is the same fix used locally for sm_120 compatibility
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/nightly/cu128",
        pre=True,
    )
    # Install all other dependencies
    .pip_install(
        "fastapi",
        "uvicorn",
        "pdfplumber",
        "easyocr",
        "pillow",
        "numpy",
        "python-dotenv",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "sentence-transformers",
        "chromadb",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-chroma",
        "langchain-huggingface",
        "langchain-text-splitters",
        "pydantic",
        "python-multipart",    # ← add this line

    )
)

# --------------------------------------------------------------------------
# Modal app definition
# --------------------------------------------------------------------------

# Create the Modal app — this is the entry point Modal uses
# All functions decorated with @app.function() run on Modal's infrastructure
app = modal.App("tldr-bot", image=image)

# Persistent volumes — these survive container restarts
# Same as Docker volume mounts but managed by Modal
# chroma_db stores document chunks between requests
# chat_history stores SQLite messages between requests
chroma_volume = modal.Volume.from_name("tldr-bot-chroma", create_if_missing=True)
db_volume = modal.Volume.from_name("tldr-bot-db", create_if_missing=True)

# Modal secrets — stores your HuggingFace token securely
# Create this in Modal dashboard: modal.com → Secrets → New Secret
# Name it "huggingface" with key HF_TOKEN = your token
# Never hardcode tokens in code
hf_secret = modal.Secret.from_name("huggingface")

# --------------------------------------------------------------------------
# The FastAPI app — same logic as api.py but decorated for Modal
# @app.function tells Modal to run this on their infrastructure
# gpu="A10G" requests an NVIDIA A10G (24GB VRAM) — enough for 7B easily
# volumes mounts our persistent storage into the container
# secrets injects HF_TOKEN as an environment variable
# timeout=600 allows up to 10 minutes for long document summarization
# container_idle_timeout=300 keeps container warm for 5 minutes after last request
#   so back-to-back requests don't each pay cold start cost
# --------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    volumes={
        "/chroma_db": chroma_volume,
        "/chat_history": db_volume,
    },
    secrets=[hf_secret],
    timeout=600,
    scaledown_window=300,
)


@modal.asgi_app()
def fastapi_app():
    # All imports inside the function so they run inside the Modal container
    import torch
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from dotenv import load_dotenv
    from huggingface_hub import login
    from peft import PeftModel
    from PIL import Image
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TextIteratorStreamer
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    import easyocr
    import numpy as np
    import pdfplumber
    import sqlite3

    # --------------------------------------------------------------------------
    # Config — same values as api.py
    # --------------------------------------------------------------------------

    HF_TOKEN     = os.environ["HF_TOKEN"]
    BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
    ADAPTER_REPO = "DraSlayer/personal-v2-llm-phase8-7b"

    MAX_CHAT_TOKENS       = 350
    MAX_SUMMARY_TOKENS    = 3000
    MAX_INPUT_TOKENS      = 30000
    LONG_DOC_CHUNK_TOKENS = 10000

    # Paths inside the Modal container — these map to our persistent volumes
    DB_PATH     = "/chat_history/chat_history.db"
    CHROMA_PATH = "/chroma_db"

    # --------------------------------------------------------------------------
    # FastAPI app
    # --------------------------------------------------------------------------

    web_app = FastAPI(title="TLDR BOT API")
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class DisableCSRF(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.headers.__dict__["_list"] = [
                (k, v) for k, v in request.headers.raw
                if k.lower() != b"origin"
            ]
            return await call_next(request)

    web_app.add_middleware(DisableCSRF)

    from starlette.middleware.trustedhost import TrustedHostMiddleware

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions based on the provided context.

If asked about a specific book, film, TV show, person, statistic, survey, or event that you cannot verify exists in your knowledge, say clearly that you cannot find reliable information about it rather than generating a plausible-sounding description. Never fabricate plot details, biographies, statistics, or historical specifics you cannot verify.

Always base your answers on the context provided to you."""

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
"Medium": """Read this document and list all the important concepts.

For each concept write:
1. What it is (2-3 sentences)
2. Example: [a concrete example]

Do not skip the Example section for any concept.

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
    # SQLite setup — same functions as api.py
    # --------------------------------------------------------------------------

    def init_db():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT     NOT NULL,
                    role       TEXT     NOT NULL,
                    content    TEXT     NOT NULL,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def db_save_message(session_id: str, role: str, content: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def db_load_messages(session_id: str) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def db_clear_messages(session_id: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    init_db()

    # --------------------------------------------------------------------------
    # Load models
    # --------------------------------------------------------------------------

    ocr_reader = easyocr.Reader(["en"], gpu=False)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    login(token=HF_TOKEN)

    torch.cuda.empty_cache()
    gc.collect()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )

    torch.cuda.empty_cache()
    gc.collect()

    llm = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
    llm.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Model ready: {ADAPTER_REPO}")

    # --------------------------------------------------------------------------
    # LangChain setup
    # --------------------------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name="documents",
    )

    session_memories: dict[str, InMemoryChatMessageHistory] = {}

    def get_memory(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_memories:
            memory = InMemoryChatMessageHistory()
            for msg in db_load_messages(session_id):
                if msg["role"] == "user":
                    memory.add_message(HumanMessage(content=msg["content"]))
                else:
                    memory.add_message(AIMessage(content=msg["content"]))
            session_memories[session_id] = memory
        return session_memories[session_id]

    # --------------------------------------------------------------------------
    # Pydantic models
    # --------------------------------------------------------------------------

    class SummarizeRequest(BaseModel):
        text: str
        length: str

    class SummarizeResponse(BaseModel):
        summary: str

    class ChatRequest(BaseModel):
        question: str
        session_id: str

    class ChatResponse(BaseModel):
        answer: str

    class StoreRequest(BaseModel):
        filename: str
        text: str
        session_id: str

    class StoreResponse(BaseModel):
        chunks_stored: int

    # --------------------------------------------------------------------------
    # Helper functions — same as api.py
    # --------------------------------------------------------------------------

    def truncate_text(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        truncated_ids = token_ids[:max_tokens]
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        print(f"[truncate_text] Input was {len(token_ids)} tokens — truncated to {max_tokens}")
        return truncated_text

    def generate(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            output = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # ← add this
            )
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return full[len(prompt_text):].strip()

    def generate_stream(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # ← add this

        )
        thread = Thread(target=llm.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    def summarize_long_document(text: str, length: str) -> str:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            tokenizer.decode(token_ids[i:i + LONG_DOC_CHUNK_TOKENS], skip_special_tokens=True)
            for i in range(0, len(token_ids), LONG_DOC_CHUNK_TOKENS)
        ]
        print(f"[long doc] {len(token_ids)} tokens split into {len(chunks)} chunks")
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"[long doc] Summarizing chunk {i + 1}/{len(chunks)}")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Summarize the key points from this section of a document in 3-5 bullet points. "
                    f"Be concise — each bullet should be one clear sentence.\n\nSection {i + 1}:\n{chunk}"
                )},
            ]
            chunk_summary = generate(messages, max_new_tokens=500)
            chunk_summaries.append(f"Section {i + 1}:\n{chunk_summary}")
        combined = "\n\n".join(chunk_summaries)
        prompt_template = SUMMARY_PROMPTS.get(length, SUMMARY_PROMPTS["Medium"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_template.format(text=combined)},
        ]
        return generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS)

    # --------------------------------------------------------------------------
    # Endpoints — same as api.py
    # --------------------------------------------------------------------------

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.get("/history")
    async def get_history(session_id: str):
        messages = db_load_messages(session_id)
        return {"messages": messages}

    @web_app.delete("/history")
    async def clear_history(session_id: str):
        db_clear_messages(session_id)
        if session_id in session_memories:
            session_memories[session_id].clear()
        return {"status": "cleared"}

    @web_app.post("/extract")
    async def extract(file: UploadFile = File(...)):
        filename = file.filename.lower()
        contents = await file.read()
        if filename.endswith(".pdf"):
            try:
                with pdfplumber.open(io.BytesIO(contents)) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                return {"text": "\n\n".join(pages_text)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(io.BytesIO(contents))
                image_np = np.array(image)
                results = ocr_reader.readtext(image_np)
                extracted = " ".join([block[1] for block in results])
                return {"text": extracted}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image OCR failed: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

    @web_app.post("/store", response_model=StoreResponse)
    async def store(request: StoreRequest):
        existing = vector_store.get(where={"$and": [{"filename": request.filename}, {"session_id": request.session_id}]})
        if existing and existing.get("ids"):
            return StoreResponse(chunks_stored=0)
        docs = text_splitter.create_documents(
            texts=[request.text],
            metadatas=[{"filename": request.filename, "session_id": request.session_id}],
        )
        vector_store.add_documents(docs)
        return StoreResponse(chunks_stored=len(docs))

    @web_app.post("/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])
        safe_text = truncate_text(request.text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_template.format(text=safe_text)},
        ]
        summary = generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS)
        return SummarizeResponse(summary=summary)

    @web_app.post("/summarize/stream")
    async def summarize_stream(request: SummarizeRequest):
        token_count = len(tokenizer.encode(request.text, add_special_tokens=False))
        if token_count <= MAX_INPUT_TOKENS:
            prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])
            safe_text = truncate_text(request.text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_template.format(text=safe_text)},
            ]
            return StreamingResponse(
                generate_stream(messages, max_new_tokens=MAX_SUMMARY_TOKENS),
                media_type="text/plain",
            )
        else:
            num_chunks = (token_count + LONG_DOC_CHUNK_TOKENS - 1) // LONG_DOC_CHUNK_TOKENS

            def long_doc_stream():
                yield f"📄 Long document detected ({token_count:,} tokens, ~{num_chunks} sections).\n"
                yield f"Summarizing each section then combining — this may take a few minutes...\n\n"
                summary = summarize_long_document(request.text, request.length)
                yield summary

            return StreamingResponse(long_doc_stream(), media_type="text/plain")

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        memory = get_memory(request.session_id)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "filter": {"session_id": request.session_id}}
        )
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
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
        answer = generate(messages, max_new_tokens=MAX_CHAT_TOKENS)
        memory.add_message(HumanMessage(content=request.question))
        memory.add_message(AIMessage(content=answer))
        db_save_message(request.session_id, "user", request.question)
        db_save_message(request.session_id, "assistant", answer)
        return ChatResponse(answer=answer)

    @web_app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        memory = get_memory(request.session_id)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "filter": {"session_id": request.session_id}}
        )
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
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
            full_answer = ""
            for token in generate_stream(messages, max_new_tokens=MAX_CHAT_TOKENS):
                full_answer += token
                yield token
            memory.add_message(HumanMessage(content=request.question))
            memory.add_message(AIMessage(content=full_answer))
            db_save_message(request.session_id, "user", request.question)
            db_save_message(request.session_id, "assistant", full_answer)

        return StreamingResponse(stream_and_save(), media_type="text/plain")

    return web_app