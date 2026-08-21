"""
TLDR BOT — Modal Deployment
9B Phase 18 model with standing cascade system prompt (reconstructed from
phase8b_summarization_diagnostic_results.json evidence) and USE_SYSTEM_PROMPT
toggle for A/B testing.

What changed from previous version:
  - SYSTEM_PROMPT added back, reconstructed from diagnostic base_response traces
  - USE_SYSTEM_PROMPT toggle — flip to False + redeploy to A/B test without it
  - build_messages() helper centralizes prompt construction so every endpoint
    (chat, summarize, long-doc chunk summaries) gets identical treatment
  - ADAPTER_REPO corrected to DraSlayer/personal-llm-phase18-9b (verified from
    the actual cascade doc, not a guessed name)
  - Everything else unchanged: enable_thinking=False + strip_thinking() fallback,
    format_summary_output() for Example: line breaks, timeout=900,
    scaledown_window=60, heartbeat yields in summarize_long_document(),
    StopOnRepetition stopping criteria, clean_text() for Cengage/URL stripping

Deploy with: modal deploy modal_app.py
"""

import io
import gc
import os
import re
import sqlite3
from threading import Thread

import modal

# --------------------------------------------------------------------------
# Modal image
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
    timeout=900,
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

    # --------------------------------------------------------------------------
    # Config — 9B Phase 18
    # --------------------------------------------------------------------------

    HF_TOKEN     = os.environ["HF_TOKEN"]
    BASE_MODEL   = "Qwen/Qwen3.5-9B"                      # confirm this matches your exact base
    ADAPTER_REPO = "DraSlayer/personal-llm-phase18-9b"    # verified from cascade doc

    MAX_CHAT_TOKENS       = 350
    MAX_SUMMARY_TOKENS    = 3000
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

    # --------------------------------------------------------------------------
    # Standing cascade system prompt
    # Reconstructed from phase8b_summarization_diagnostic_results.json —
    # the diagnostic's base_response reasoning traces explicitly quote this
    # exact system prompt during evaluation, and the Phase 18 adapter_response
    # quality (clean, concise, no <think> leakage) suggests later phases were
    # trained WITH it present. USE_SYSTEM_PROMPT lets you A/B test this.
    # --------------------------------------------------------------------------

    USE_SYSTEM_PROMPT = True

    SYSTEM_PROMPT = """When writing authentication or credential-checking code, always verify passwords via proper hash verification, not plaintext comparison.

If a question states or assumes something technically false (a wrong complexity/Big-O claim, a wrong algorithmic property, or any other incorrect premise), correct the false premise directly before answering, even if the rest of your answer is brief. Do not accept an incorrect framing at face value just to keep the response short."""

    def build_messages(user_content: str) -> list:
        """
        Wraps a user message with the standing system prompt if enabled.
        Centralized so every call site (chat, summarize, long-doc chunk
        summaries) gets identical treatment — avoids prompt drift.
        """
        if USE_SYSTEM_PROMPT:
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        return [{"role": "user", "content": user_content}]

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

        "Medium": """Read this document and explain the key concepts in your own words — do not copy slide text or headings verbatim.

For each concept use exactly this format:

**[Concept Name]**
What it is: [2-3 sentences explaining it clearly]
Example: [a concrete, specific example — use real numbers, code, or details from the document where possible]

Put a blank line between "What it is" and "Example", and a blank line before each new concept.

Cover every distinct concept in the document, not just the first few.

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
3-5 things most likely to be tested, ranked by likelihood.

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
    print(f"System prompt enabled: {USE_SYSTEM_PROMPT}")

    # --------------------------------------------------------------------------
    # Stopping criteria — catches hard repetition loops
    # --------------------------------------------------------------------------

    class StopOnRepetition(StoppingCriteria):
        def __init__(self, threshold=15):
            self.threshold = threshold

        def __call__(self, input_ids, scores, **kwargs):
            last_tokens = input_ids[0][-self.threshold:].tolist()
            return len(set(last_tokens)) == 1

    stopping_criteria = StoppingCriteriaList([StopOnRepetition(threshold=15)])

    # --------------------------------------------------------------------------
    # Output post-processing
    # --------------------------------------------------------------------------

    def strip_thinking(text: str) -> str:
        # Qwen3.5 wraps reasoning in <think>...</think> before the real answer
        # enable_thinking=False should prevent this at the source — this is
        # a safety net in case it still appears
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def format_summary_output(text: str) -> str:
        # Forces "Example:" onto its own line with a blank line before it
        text = re.sub(r'(?<!\n\n)Example:', r'\n\nExample:', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

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
        text = re.sub(
            r'@\d{4}\s*Cengage\..*?classroom use\.',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r'http\S+|www\.\S+', '', text)
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
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        raw = strip_thinking(raw)
        raw = format_summary_output(raw)
        return raw

    def generate_stream(messages: list, max_new_tokens: int = MAX_CHAT_TOKENS):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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

    def summarize_long_document(text: str, length: str):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            tokenizer.decode(token_ids[i:i + LONG_DOC_CHUNK_TOKENS], skip_special_tokens=True)
            for i in range(0, len(token_ids), LONG_DOC_CHUNK_TOKENS)
        ]
        print(f"[long doc] {len(token_ids)} tokens split into {len(chunks)} chunks")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            yield f"⏳ Processing section {i + 1}/{len(chunks)}...\n"
            print(f"[long doc] Summarizing chunk {i + 1}/{len(chunks)}")
            messages = build_messages(
                f"Summarize the key points from section {i + 1} in 3-5 bullet points. "
                f"Each bullet should be one clear sentence.\n\n{chunk}"
            )
            chunk_summary = generate(messages, max_new_tokens=300)
            chunk_summaries.append(f"Section {i + 1}:\n{chunk_summary}")

        combined = "\n\n".join(chunk_summaries)
        safe_combined = truncate_text(combined, max_tokens=15000)
        prompt_template = SUMMARY_PROMPTS.get(length, SUMMARY_PROMPTS["Medium"])
        messages = build_messages(prompt_template.format(text=safe_combined))
        yield "\n📝 Combining into final summary...\n\n"
        yield generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS)

    # --------------------------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------------------------

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "system_prompt_enabled": USE_SYSTEM_PROMPT}

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
                return {"text": cleaned_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
        elif filename.endswith((".md", ".txt")):
            try:
                text = contents.decode("utf-8")
                cleaned = clean_text(text)
                return {"text": cleaned}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Text file parsing failed: {str(e)}")
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
        messages = build_messages(prompt_template.format(text=safe_text))
        return SummarizeResponse(summary=generate(messages, max_new_tokens=MAX_SUMMARY_TOKENS))

    @web_app.post("/summarize/stream")
    async def summarize_stream(request: SummarizeRequest):
        token_count = len(tokenizer.encode(request.text, add_special_tokens=False))
        print(f"[summarize] Token count: {token_count}")

        if token_count <= MAX_INPUT_TOKENS:
            prompt_template = SUMMARY_PROMPTS.get(request.length, SUMMARY_PROMPTS["Medium"])
            safe_text = truncate_text(request.text)
            messages = build_messages(prompt_template.format(text=safe_text))
            return StreamingResponse(
                generate_stream(messages, max_new_tokens=MAX_SUMMARY_TOKENS),
                media_type="text/plain",
            )
        else:
            num_chunks = (token_count + LONG_DOC_CHUNK_TOKENS - 1) // LONG_DOC_CHUNK_TOKENS
            print(f"[summarize] Long doc: {token_count} tokens → {num_chunks} chunks")

            def long_doc_stream():
                yield f"📄 Long document detected ({token_count:,} tokens, ~{num_chunks} sections).\n\n"
                yield from summarize_long_document(request.text, request.length)

            return StreamingResponse(long_doc_stream(), media_type="text/plain")

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        memory = get_memory(request.session_id)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2, "filter": {"session_id": request.session_id}}
        )
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        recent_messages = memory.messages[-4:] if len(memory.messages) > 4 else memory.messages
        history = "\n".join([
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in recent_messages
        ])

        user_content = (
            (f"Earlier in our conversation:\n{history}\n\n" if history else "") +
            f"Using the document excerpts below, answer this question: {request.question}\n\n"
            f"Document excerpts:\n{context}"
        )
        messages = build_messages(user_content)

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

        recent_messages = memory.messages[-4:] if len(memory.messages) > 4 else memory.messages
        history = "\n".join([
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in recent_messages
        ])

        user_content = (
            (f"Earlier in our conversation:\n{history}\n\n" if history else "") +
            f"Using the document excerpts below, answer this question: {request.question}\n\n"
            f"Document excerpts:\n{context}"
        )
        messages = build_messages(user_content)

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