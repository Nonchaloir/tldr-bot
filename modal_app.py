"""
TLDR BOT — Modal Deployment
Final stable version.

What changed from previous version:
  - StopOnRepetition stopping criteria added back — catches hard loops
    (same token 15 times) without distorting normal output
  - No repetition_penalty, no no_repeat_ngram_size — matches eval script
  - eos_token_id explicitly set so model stops naturally when done
  - stopping_criteria catches cases where eos token isn't generated cleanly
  - clean_text() strips Cengage copyright and URLs from PDF
  - No system prompt — matches eval script setup
  - scaledown_window=300 — stays warm 5 minutes to reduce cold starts
  - Debug prints kept for Modal logs investigation

Deploy with: modal deploy modal_app.py
"""

import io
import gc
import os
import sqlite3
from threading import Thread

import modal

# --------------------------------------------------------------------------
# Modal image — pinned package versions to match local environment
# --------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/nightly/cu128",
        pre=True,
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pdfplumber",
        "easyocr",
        "pillow",
        "numpy",
        "python-dotenv",
        "transformers==5.13.1",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "bitsandbytes==0.49.2",
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
        "starlette",
    )
)

# --------------------------------------------------------------------------
# Modal app definition
# --------------------------------------------------------------------------

app = modal.App("tldr-bot", image=image)

chroma_volume = modal.Volume.from_name("tldr-bot-chroma", create_if_missing=True)
db_volume     = modal.Volume.from_name("tldr-bot-db",     create_if_missing=True)
hf_secret     = modal.Secret.from_name("huggingface")

@app.function(
    gpu="A10G",
    volumes={
        "/chroma_db":    chroma_volume,
        "/chat_history": db_volume,
    },
    secrets=[hf_secret],
    timeout=600,
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    import torch
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from huggingface_hub import login
    from peft import PeftModel
    from PIL import Image
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TextIteratorStreamer,
        StoppingCriteria, StoppingCriteriaList,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    import easyocr
    import numpy as np
    import pdfplumber
    import re

    # --------------------------------------------------------------------------
    # Config
    # --------------------------------------------------------------------------

    HF_TOKEN     = os.environ["HF_TOKEN"]
    BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_REPO = "DraSlayer/personal-llm-phase17-3b"

    # Matched to eval script — 350 was used in phase 17 eval
    # Model stops naturally via eos_token_id before hitting this ceiling
    # This is a safety net, not a target
    MAX_CHAT_TOKENS       = 350
    MAX_SUMMARY_TOKENS    = 1500
    MAX_INPUT_TOKENS      = 30000
    LONG_DOC_CHUNK_TOKENS = 10000

    DB_PATH     = "/chat_history/chat_history.db"
    CHROMA_PATH = "/chroma_db"

    # --------------------------------------------------------------------------
    # FastAPI app
    # --------------------------------------------------------------------------

    web_app = FastAPI(title="TLDR BOT API")

    class DisableCSRF(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.headers.__dict__["_list"] = [
                (k, v) for k, v in request.headers.raw
                if k.lower() != b"origin"
            ]
            return await call_next(request)

    web_app.add_middleware(DisableCSRF)
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # No system prompt — model was trained and evaluated without one
    # Adding a system prompt changes the input distribution and can cause
    # the model to output training artifacts instead of natural responses

    SUMMARY_PROMPTS = {
        "Short": """Analyse this document and produce a concise breakdown.

For each of the 2-3 most important concepts write:
**[Concept Name]**
→ What it is: one clear sentence
→ Why it matters: one sentence

End with:
**TL;DR:** One paragraph summary in simple language.

Document:
{text}""",

"Medium": """The following is a university lecture document. Extract and explain all the key concepts from it.

Do not ask questions. Do not say you can help. Just directly output the concepts.

For each concept use this format:
**[Concept Name]**
What it is: [2 sentences]
Example: [one example]

---

Document:
{text}""",
        "Detailed": """Analyse this document and produce an in-depth study breakdown.

For each major concept write:

**[Number]. [Concept Name]**
→ **What it is:** 3-5 sentences covering what it means and how it works.
→ **Key details:** Important conditions, formulas, or rules from the document.
→ **Example:** A worked example showing the concept in action.

End with:

**TL;DR**
4-6 sentences summarising the document in plain simple language.

**Likely Exam Topics**
3-5 things most likely to be tested.

Document:
{text}""",
    }

    # --------------------------------------------------------------------------
    # SQLite setup
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
    # Stopping criteria
    # Catches hard repetition loops — same token 15 times in a row
    # This is different from no_repeat_ngram_size which penalizes ALL repetition
    # and can distort natural academic language that repeats certain phrases
    # StopOnRepetition only fires on extreme degeneration (identical tokens)
    # leaving normal varied output completely untouched
    # --------------------------------------------------------------------------

    class StopOnRepetition(StoppingCriteria):
        def __init__(self, threshold=25):
            self.threshold = threshold

        def __call__(self, input_ids, scores, **kwargs):
            # If last 15 tokens are all the same — degeneration, stop immediately
            last_tokens = input_ids[0][-self.threshold:].tolist()
            return len(set(last_tokens)) == 1

    stopping_criteria = StoppingCriteriaList([StopOnRepetition(threshold=15)])

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
    # Helper functions
    # --------------------------------------------------------------------------

    def clean_text(text: str) -> str:
        # Remove Cengage copyright notice from every slide bottom
        # This was appearing in every ChromaDB chunk and triggering hallucinations
        text = re.sub(
            r'@\d{4}\s*Cengage\..*?classroom use\.',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # Remove URLs — trigger model to hallucinate web content
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Clean up extra whitespace left after removals
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def truncate_text(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        truncated_ids = token_ids[:max_tokens]
        print(f"[truncate_text] Input was {len(token_ids)} tokens — truncated to {max_tokens}")
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def generate(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS) -> str:
        # Matches eval script exactly
        # eos_token_id — stops when model naturally finishes
        # stopping_criteria — catches hard loops that eos misses
        # No repetition_penalty or no_repeat_ngram_size — matches eval script
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        # Slice new tokens only — same as eval script
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_stream(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS):
        # Matches eval script as closely as possible for streaming
        # eos_token_id — stops naturally
        # stopping_criteria — catches hard loops
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(llm.device)
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
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
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
                {"role": "user", "content": (
                    f"Summarize the key points from section {i + 1} in 3-5 bullet points. "
                    f"Each bullet should be one clear sentence.\n\n{chunk}"
                )},
            ]
            chunk_summary = generate(messages, max_new_tokens=300)
            chunk_summaries.append(f"Section {i + 1}:\n{chunk_summary}")

        combined = "\n\n".join(chunk_summaries)
        safe_combined = truncate_text(combined, max_tokens=15000)
        prompt_template = SUMMARY_PROMPTS.get(length, SUMMARY_PROMPTS["Medium"])
        messages = [
            {"role": "user", "content": prompt_template.format(text=safe_combined)},
        ]
        return generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS)

    # --------------------------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------------------------

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.get("/history")
    async def get_history(session_id: str):
        return {"messages": db_load_messages(session_id)}

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
                raw_text = "\n\n".join(pages_text)
                cleaned_text = clean_text(raw_text)
                print(f"[extract] Raw: {len(raw_text)} chars → Cleaned: {len(cleaned_text)} chars")
                print(f"[extract] Preview:\n{cleaned_text[:1000]}")
                return {"text": cleaned_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(io.BytesIO(contents))
                image_np = np.array(image)
                results = ocr_reader.readtext(image_np)
                return {"text": " ".join([block[1] for block in results])}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image OCR failed: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

    @web_app.post("/store", response_model=StoreResponse)
    async def store(request: StoreRequest):
        existing = vector_store.get(where={"session_id": request.session_id})
        if existing and existing.get("metadatas"):
            filenames = [m.get("filename") for m in existing["metadatas"]]
            if request.filename in filenames:
                print(f"[store] {request.filename} already stored, skipping")
                return StoreResponse(chunks_stored=0)
        docs = text_splitter.create_documents(
            texts=[request.text],
            metadatas=[{"filename": request.filename, "session_id": request.session_id}],
        )
        vector_store.add_documents(docs)
        print(f"[store] Stored {len(docs)} chunks for {request.filename}")
        return StoreResponse(chunks_stored=len(docs))

    @web_app.post("/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])
        safe_text = truncate_text(request.text)
        messages = [{"role": "user", "content": prompt_template.format(text=safe_text)}]
        return SummarizeResponse(summary=generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS))

    @web_app.post("/summarize/stream")
    async def summarize_stream(request: SummarizeRequest):
        token_count = len(tokenizer.encode(request.text, add_special_tokens=False))
        print(f"[summarize] Token count: {token_count}")
        print(f"[summarize] Text preview:\n{request.text[:500]}")

        if token_count <= MAX_INPUT_TOKENS:
            prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])
            safe_text = truncate_text(request.text)
            messages = [{"role": "user", "content": prompt_template.format(text=safe_text)}]
            return StreamingResponse(
                generate_stream(messages, max_new_tokens=MAX_SUMMARY_TOKENS),
                media_type="text/plain",
            )
        else:
            num_chunks = (token_count + LONG_DOC_CHUNK_TOKENS - 1) // LONG_DOC_CHUNK_TOKENS
            print(f"[summarize] Long doc: {token_count} tokens → {num_chunks} chunks")

            def long_doc_stream():
                yield f"📄 Long document detected ({token_count:,} tokens, ~{num_chunks} sections).\n"
                yield f"Summarizing each section then combining — this may take a few minutes...\n\n"
                yield summarize_long_document(request.text, request.length)

            return StreamingResponse(long_doc_stream(), media_type="text/plain")

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        memory = get_memory(request.session_id)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2, "filter": {"session_id": request.session_id}}
        )
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        print(f"[chat] Question: {request.question}")
        print(f"[chat] Context preview:\n{context[:500]}")
        print(f"[chat] History messages: {len(memory.messages)}")

        recent_messages = memory.messages[-4:] if len(memory.messages) > 4 else memory.messages
        history = "\n".join([
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in recent_messages
        ])

        messages = [
            {"role": "user", "content": (
                (f"Earlier in our conversation:\n{history}\n\n" if history else "") +
                f"Using the document excerpts below, answer this question: {request.question}\n\n"
                f"Document excerpts:\n{context}"
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
            search_kwargs={"k": 2, "filter": {"session_id": request.session_id}}
        )
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        print(f"[chat/stream] Question: {request.question}")
        print(f"[chat/stream] Context preview:\n{context[:500]}")
        print(f"[chat/stream] History messages: {len(memory.messages)}")

        recent_messages = memory.messages[-4:] if len(memory.messages) > 4 else memory.messages
        history = "\n".join([
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in recent_messages
        ])

        messages = [
            {"role": "user", "content": (
                (f"Earlier in our conversation:\n{history}\n\n" if history else "") +
                f"Using the document excerpts below, answer this question: {request.question}\n\n"
                f"Document excerpts:\n{context}"
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