import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage
import json

from langgraph.checkpoint.sqlite import SqliteSaver

from pydantic import BaseModel, Field
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_core.tools import tool
# data class for user ids
from dataclasses import dataclass
# for context saving
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()



# 1. Initialize the backend endpoint
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.3-70B-Instruct",
#     task="conversational",
#     temperature=0.1,
#     max_new_tokens=1024
# )

# # 2. Wrap it for chat templates
# chat_model = ChatHuggingFace(llm=llm)


# # 2. Initialize the Gemini model (e.g., gemini-1.5-flash)
# llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash", temperature=0.9)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



# Replace with the path to your actual folder
folder_path = Path("documents")

INDEX_PATH = "faiss_index"          # where the vector store persists on disk
CHECKPOINT_DB = "chat_history.sqlite"  # where conversation history persists


# Create embedding model

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)


documentation = []



if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this file ourselves
    )
else:
    print("No existing index found — building a new one...")

    documentation = list(folder_path.glob("*.pdf"))
    print(f"Found PDFs: {documentation}")

    documents = []
    for document in documentation:
        loader = PyPDFLoader(str(document))  # str() avoids Path-related loader issues
        documents.extend(loader.load())

    print(f"Loaded pages: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=400,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created chunks: {len(docs)}")
    print(f"Loaded pages: {len(documents)}")

    print("Number of chunks:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\n========== CHUNK {i + 1} ==========")
        print(doc.page_content)


    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)
    print(f"Saved index to '{INDEX_PATH}/'")

# If you add/remove/change PDFs later, delete the faiss_index/ folder
# and re-run so it gets rebuilt from the updated documents.


# 7. Define structured response

# class Source(BaseModel):
#     file: str = Field(description="Name of the source PDF")
#     page: int = Field(description="Page number in the PDF")

# class RAGResponse(BaseModel):
#     answer: str = Field(
#     description="Answer to the user's question based only on retrieved context"
#     )
#     sources: List[Source] = Field(
#     description="PDF documents and pages used to answer the question"
#     )

@tool
def search_documents(query: str) -> str:
    """Search the PDF documents for information relevant to the user's query."""

    results = db.similarity_search_with_score(query, k=5)

    retrieved_documents = []

    for doc, score in results:
        retrieved_documents.append({
            "content": doc.page_content,
            "source": str(doc.metadata.get("source")),
            "page": doc.metadata.get("page", 0) + 1,
            "score": float(score)  # lower = more similar (FAISS L2 distance)
        })

    return json.dumps(retrieved_documents)


thread_config = {"configurable": {"thread_id": "1"}}
# 5. Search

query = "How much funding did the project recieve"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:

    agent = create_agent(
        model=llm,
        tools=[search_documents],
        checkpointer=checkpointer,
    )

    messages = [
        SystemMessage(content="""You are a helpful assistant.

Use the search_documents tool whenever the user asks
about information that may be contained in the documents.

Answer using only information from the retrieved documents.
Do not mention sources or filenames in your answer — that will
be handled separately. Also if requested info is not found in the documents, 
say "I could not find any information about that in the documents." and do not make up an answer.
"""),
        HumanMessage(content=query),
    ]

    response = agent.invoke(
        {"messages": messages},
        thread_config,
    )

    answer = response["messages"][-1].content

    print("\n================ ANSWER ================\n")
    print(answer)

    tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]

    # print("\n================ SOURCE ================\n")

    # if tool_messages:
    #     retrieved = json.loads(tool_messages[-1].content)
    #     if retrieved:
    #         top_match = min(retrieved, key=lambda d: d["score"])
    #         print(f"Source: {top_match['source']}, page {top_match['page']}")
    #     else:
    #         print("No documents were retrieved for this query.")
    # else:
    #     print("The agent did not call search_documents for this query.")