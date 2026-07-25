# Project Handoff — Document & Image Summarizer
## For continuing in a new chat session

---

## What This Project Is

A personal document summarizer and Q&A tool built for a university student. The user uploads lecture PDFs or screenshots, the app summarizes them in a structured breakdown with key concepts, explanations, examples, and a TL;DR. The user can then ask follow-up questions about the documents in a chat interface.

The user is also training their own LLM (Qwen2.5-7B-Instruct with LoRA adapters) in a 17-phase cascade on HuggingFace under `DraSlayer/`. The app uses their trained model for all summarization and chat.

---

## Project Stack

```
app.py        — Streamlit frontend (UI only, no AI logic)
api.py        — FastAPI backend (all AI logic lives here)
.env          — credentials and model config (never commit this)
.gitignore    — excludes .env and chroma_db/
requirements.txt — all dependencies
chroma_db/    — ChromaDB persistent storage (auto-created on first run)
```

**Run order — always start FastAPI first:**
```
Terminal 1: uvicorn api:app --reload
Terminal 2: streamlit run app.py
```

---

## Steps Completed (1-6 of 10)

### Step 1 — Python Foundations
Already knew Python. Skipped.

### Step 2 — Streamlit UI
Built the full UI shell in `app.py`:
- `st.file_uploader` — multi-file upload, server holds files between reruns
- `st.session_state` — persists `summaries` (dict) and `chat_history` (list) across reruns
- `st.chat_message` / `st.chat_input` — chat interface
- `st.expander` — collapsible sections per file
- `st.sidebar` — settings (summary length slider, clear button)
- `st.write_stream` — streams tokens word by word
- Walrus operator `:=` — assigns and checks `st.chat_input` in one line
- Streamlit reruns the ENTIRE script on every interaction — session_state is the only persistence

### Step 3 — Document Parsing
Real parsing in `extract_text()` in `api.py`:
- PDF → `pdfplumber` (digital PDFs only, not scanned)
- Images → `EasyOCR` with `gpu=False` (placeholder until Qwen VL model is ready)
- `io.BytesIO` — type mismatch fix: Streamlit gives raw bytes, libraries need file-like objects
- `easyocr.Reader(["en"], gpu=False)` — forced CPU because RTX 5070 (sm_120) is incompatible with current PyTorch stable builds

### Step 4 — FastAPI Backend
Split into two separate programs:
- `app.py` — purely UI, calls FastAPI and displays results
- `api.py` — all the real work

Endpoints:
- `GET /health` — server status check
- `POST /extract` — file → extracted text
- `POST /summarize` — text + length → summary (non-streaming fallback)
- `POST /summarize/stream` — text + length → streaming summary
- `POST /store` — text → chunked + embedded + stored in ChromaDB
- `POST /chat` — question → answer (non-streaming fallback)
- `POST /chat/stream` — question → streaming answer

Pydantic models define exact shape of data in/out of each endpoint. FastAPI validates automatically.
CORS middleware allows Streamlit (port 8501) to talk to FastAPI (port 8000).
`async def` + `await` — FastAPI can handle multiple requests concurrently.

### Step 5 — RAG + ChromaDB
When a file is uploaded and summarized:
1. `extract_text()` — get text from file
2. `store_text()` → `POST /store` — chunk text (500 chars, 50 char overlap), embed with `all-MiniLM-L6-v2`, store in ChromaDB
3. `summarize_text()` → `POST /summarize/stream` — real model summary

When user asks a question in chat:
1. Convert question to embedding
2. Query ChromaDB for top 3 most similar chunks
3. Pass chunks as context to model
4. Model answers from document context, not general knowledge

ChromaDB uses `PersistentClient(path="./chroma_db")` — saves to disk, survives server restarts.
`get_or_create_collection(name="documents")` — creates on first run, reuses on subsequent runs.
`results["documents"]` — ChromaDB always returns this fixed key regardless of collection name.
`results["documents"][0]` — `[0]` because ChromaDB returns list of lists (supports multiple queries).

Embedding model: `all-MiniLM-L6-v2` from HuggingFace — free, open source, converts text to numbers. Similar meaning = similar numbers. The user can fine-tune this later on Computer Engineering content for better domain-specific retrieval.

### Step 6 — Model Integration
Real model loaded at startup in `api.py`:
- Base: `Qwen/Qwen2.5-7B-Instruct`
- Adapter: `DraSlayer/personal-v2-llm-phase6-7b` (Phase 6 of 17-phase cascade)
- QLoRA 4-bit quantization (NF4) — fits 7B model into 8GB VRAM
- `device_map="auto"` — lets PyTorch choose GPU/CPU (matches eval script pattern)
- `PeftModel.from_pretrained()` — stacks LoRA adapter on base model
- `llm.eval()` — switches off dropout for deterministic inference
- To update to a new phase: just change `ADAPTER_REPO` in `.env`

Token limits:
- `MAX_CHAT_TOKENS = 350` — chat answers are concise
- `MAX_SUMMARY_TOKENS = 1500` — summaries need room for full breakdowns
- `MAX_INPUT_TOKENS = 30000` — model context window limit (32,768 total, reserve 2,000 for overhead)

`truncate_text()` — tokenizes input and truncates if over limit. Uses actual tokenizer not character count because 1 token ≠ 1 character.

`generate()` — non-streaming. Formats messages with `apply_chat_template`, tokenizes, runs `llm.generate()`, strips prompt from output, returns string.

`generate_stream()` — streaming. Uses `TextIteratorStreamer` (acts as a queue) + background `Thread` (because `llm.generate()` is blocking). Yields tokens one by one as they arrive.

`StreamingResponse` (FastAPI) — wraps the generator and sends tokens over HTTP as they arrive.
`st.write_stream()` (Streamlit) — reads the stream and displays tokens word by word. Returns full text when done (saved to session_state).

System prompt added to mitigate known Phase 1 hallucination limitation (model fabricates fictional content). Full fix planned for Phase 15 DPO.

Structured summary prompts — three levels (Short/Medium/Detailed):
- Short: 2-3 concepts, what it is, why it matters, TL;DR
- Medium: 3-5 concepts, explanation, example, TL;DR
- Detailed: all concepts, full explanation, key details, worked example, TL;DR + Likely Exam Topics

---

## Current File Contents

### `.env`
```
HF_TOKEN=your_token_here
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
ADAPTER_REPO=DraSlayer/personal-v2-llm-phase6-7b
```

### `requirements.txt`
```
--extra-index-url https://download.pytorch.org/whl/cu124
torch
torchvision
torchaudio
streamlit
pdfplumber
easyocr
pillow
numpy
fastapi
uvicorn
requests
pydantic
chromadb
sentence-transformers
python-dotenv
transformers
peft
accelerate
bitsandbytes
huggingface_hub
```

### `.gitignore`
```
.env
chroma_db/
__pycache__/
*.pyc
```

---

## Known Issues / Hardware Notes

**RTX 5070 Laptop GPU (sm_120) incompatibility:**
- PyTorch stable (cu124) doesn't support sm_120 (Blackwell architecture)
- Fix: install PyTorch nightly with cu128 support:
  ```
  pip uninstall torch torchvision torchaudio bitsandbytes -y
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  pip install bitsandbytes --upgrade
  ```
- EasyOCR workaround: `easyocr.Reader(["en"], gpu=False)` — forced CPU since it's a placeholder anyway
- `device_map="auto"` used instead of `device_map={"": 0}` to match eval script behavior

**8GB VRAM is tight:**
- Model uses ~7.7GB with 4-bit quantization
- Only ~300MB headroom — may get OOM errors on very long generations
- Reduce `MAX_SUMMARY_TOKENS` to 400 for local testing, restore to 1500 for real use

**duplicate chunks in ChromaDB:**
- `st.session_state.summaries` resets when browser closes
- On next session, guard `if f.name not in st.session_state.summaries` won't work
- Files get re-stored in ChromaDB with duplicate chunks
- Fix (not yet implemented): check ChromaDB before storing:
  ```python
  existing = collection.get(where={"filename": request.filename})
  if existing["ids"]:
      return StoreResponse(chunks_stored=0)
  ```

---

## The User's LLM Cascade (17 phases)

The user is building a personal AI assistant on Qwen2.5-7B-Instruct using sequential LoRA adapter stacking. Each phase trains on top of the previous phase's adapter.

| Phase | Focus | Status |
|---|---|---|
| 1 | General Intelligence, premise verification, hallucination resistance | ✅ Complete |
| 2 | Chain of Thought + Math | ✅ Complete |
| 3 | Logical Reasoning | ✅ Complete |
| 4 | Math | ✅ Complete |
| 5 | Numerical Reasoning | ✅ Complete |
| 6 | Coding | ✅ Complete — current adapter |
| 7 | SQL + Tables | Next |
| 8 | Summarization | Planned |
| 9 | Long Documents | Planned |
| 10 | Fact Checking | Planned |
| 11 | NER | Planned |
| 12 | Instruction Following | Planned |
| 13 | Multi-turn Chat | Planned |
| 14 | Causal Reasoning | Planned |
| 15 | Hallucination DPO | Planned (fixes Phase 1 hallucination) |
| 16 | Cybersecurity | Planned |
| 17 | Formal Logic | Planned |

Training platform: Thunder Compute (RTX A6000, $0.35/hr)
HuggingFace org: `DraSlayer/`

Known limitations deferred to later phases:
- Hallucination on fictional/unknown content → Phase 15 DPO
- Social pressure capitulation → Phase 13 + Phase 15 DPO
- Double negative verbal framing → Phase 15 DPO
- Interest formula lecture mode → Phase 15 DPO

---

## Comment Convention

The user writes their own understanding comments with `##` (double hash).
Claude's explanatory comments use `#` (single hash).
This distinction must be preserved whenever files are updated.
Always ask the user to paste their latest commented version before updating a file — the output file loses their comments otherwise.

---

## Roadmap — Steps Remaining

From the original 10-step roadmap:

| Step | Topic | Status |
|---|---|---|
| 1 | Python Foundations | ✅ |
| 2 | Streamlit | ✅ |
| 3 | Document Parsing | ✅ |
| 4 | FastAPI | ✅ |
| 5 | RAG + ChromaDB | ✅ |
| 6 | Model Integration | ✅ |
| 7 | LangChain (Optional) | Next |
| 8 | SQLite (Chat History) | Planned |
| 9 | Docker (Deployment) | Planned |
| 10 | Git + GitHub | Planned |

**Next step is Step 7 — LangChain (optional but powerful).**
From the roadmap:
- LLM chains
- Document loaders
- Text splitters
- RetrievalQA chain
- Memory (conversation history)
- Makes RAG simpler with half the code

**Deployment plan:** Modal (free tier — $30/month compute credits, no credit card required, scales to zero when idle). Only change needed when deploying: update `API_URL` in `app.py` from `http://localhost:8000` to the Modal endpoint URL.

---

## Key Concepts the User Now Understands

- Streamlit rerun model and why session_state exists
- `with` containers (st.chat_message, st.expander)
- Walrus operator `:=`
- `io.BytesIO` as a type mismatch fix (bytes → file-like object)
- `@st.cache_resource` vs module-level variables for model caching
- FastAPI async/await and why it matters
- Pydantic models as incoming/outgoing contracts
- CORS and why it's needed between ports
- Embeddings — text converted to numbers, similar meaning = similar numbers
- ChromaDB as a meaning-based search database (vs keyword search)
- Chunking with overlap
- RAG flow: extract → chunk → embed → store → query → context → answer
- QLoRA 4-bit quantization (NF4, bfloat16 compute, double quant)
- LoRA adapter stacking on top of base model
- `generate()` vs `generate_stream()` and why Threading is needed for streaming
- `TextIteratorStreamer` as a queue between model thread and HTTP response
- `StreamingResponse` (FastAPI) + `st.write_stream()` (Streamlit) working together
- Token limits and why they matter
- `truncate_text()` using actual tokenizer not character count
