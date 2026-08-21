import json
from pathlib import Path
from typing import List, Annotated

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

import vectorstore
from ingestion import save_uploaded_files, process_documents
from agent import agent, SYSTEM_PROMPT
from models import ChatRequest, ChatResponse

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_pdfs(files: Annotated[List[UploadFile], File(...)]):
    saved_paths = save_uploaded_files(files)
    new_chunks = process_documents(saved_paths)

    if new_chunks:
        vectorstore.add_documents(new_chunks)

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
    return {"status": "ok", "index_built": vectorstore.get_db() is not None}