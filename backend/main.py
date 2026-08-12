import os
import json
import shutil
import sqlite3
from pathlib import Path
from typing import List, Optional, Annotated

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("documents")
UPLOAD_DIR.mkdir(exist_ok=True)

INDEX_PATH = "faiss_index"
CHECKPOINT_DB = "chat_history.sqlite"

# ---------------------------------------------------------------------------
# Embeddings + vector store
# Loaded once at import time and shared across every request. `db` starts
# as None if no index exists yet — it gets created on the first /upload.
# ---------------------------------------------------------------------------

# FastEmbed runs on ONNX Runtime instead of PyTorch — PyTorch alone can eat
# 500MB-1GB+ of RAM just being imported, which blows straight through
# Render's free-tier 512MB cap. FastEmbed's default model (bge-small-en-v1.5)
# is a similarly-sized, similarly-good embedding model with a much smaller
# memory footprint.
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("No FAISS index found yet — starting empty. Upload PDFs to build one.")
    db = None

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ---------------------------------------------------------------------------
# Retrieval tool
# This closes over the module-level `db`. Python closures capture the
# *variable*, not a snapshot of its value at definition time — so once
# /upload reassigns `db`, this tool automatically sees the updated index
# on the very next call, no extra wiring needed.
# ---------------------------------------------------------------------------


@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF documents for information relevant to the user's query."""
    if db is None:
        return json.dumps([])

    results = db.similarity_search_with_score(query, k=5)
    retrieved = []
    for doc, score in results:
        retrieved.append({
            "content": doc.page_content,
            "source": str(doc.metadata.get("source")),
            "page": doc.metadata.get("page", 0) + 1,
            "score": float(score),
        })
    return json.dumps(retrieved)


# ---------------------------------------------------------------------------
# Checkpointer + agent — created once at startup, reused for every request.
#
# We open the sqlite connection directly (sqlite3.connect + SqliteSaver(conn))
# instead of the `SqliteSaver.from_conn_string(...)` context manager you used
# before. That context manager closes the connection the moment the `with`
# block ends — perfect for a one-shot script, but wrong here: a server needs
# the connection to stay open for its entire lifetime, across many requests.
# check_same_thread=False is needed because FastAPI may handle requests on
# different threads.
# ---------------------------------------------------------------------------

conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model=llm,
    tools=[search_documents],
    checkpointer=checkpointer,
)

SYSTEM_PROMPT = """You are a helpful assistant.

Use the search_documents tool whenever the user asks
about information that may be contained in the documents.

Answer using only information from the retrieved documents.
Do not mention sources or filenames in your answer — that will
be handled separately. Also if requested info is not found in the documents,
say "I could not find any information about that in the documents." and do not make up an answer.
"""

# NOTE: we send a SystemMessage on every /chat call, same as your original
# script. Since the checkpointer persists history, this means the system
# prompt technically gets appended to the stored thread every turn, not just
# the first — harmless for a learning project, but on a long-running thread
# it adds a bit of repeated context. A future improvement: only prepend the
# system message when the thread has no prior history yet.

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this before deploying anywhere public
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    answer: str
    source: Optional[str] = None


@app.post("/upload")
async def upload_pdfs(files: Annotated[List[UploadFile], File(...)]):
    global db

    saved_paths = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(dest)

    # Only load + chunk the newly uploaded files — not the whole folder.
    new_documents = []
    for path in saved_paths:
        loader = PyPDFLoader(str(path))
        new_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    new_chunks = text_splitter.split_documents(new_documents)

    if db is None:
        db = FAISS.from_documents(new_chunks, embeddings)
    else:
        db.add_documents(new_chunks)  # incremental add — leaves existing vectors untouched

    db.save_local(INDEX_PATH)

    return {
        "uploaded": [p.name for p in saved_paths],
        "chunks_added": len(new_chunks),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=req.message),
    ]

    response = agent.invoke(
        {"messages": messages},
        {"configurable": {"thread_id": req.thread_id}},
    )

    answer = response["messages"][-1].content

    tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
    source = None
    if tool_messages:
        retrieved = json.loads(tool_messages[-1].content)
        if retrieved:
            top_match = min(retrieved, key=lambda d: d["score"])
            source = f"{Path(top_match['source']).name}, page {top_match['page']}"

    return ChatResponse(answer=answer, source=source)


@app.get("/health")
async def health():
    return {"status": "ok", "index_built": db is not None}
