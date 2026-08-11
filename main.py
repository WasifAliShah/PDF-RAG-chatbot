import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage
import json

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
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()



# 1. Initialize the backend endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="conversational",
    temperature=0.1,
    max_new_tokens=1024
)

# 2. Wrap it for chat templates
chat_model = ChatHuggingFace(llm=llm)


documentation = []
# Replace with the path to your actual folder
folder_path = Path("C:\\LangChain\\Chatbot\\documents")

# Loop through all files ending in .pdf in this folder
for pdf_file in folder_path.glob("*.pdf"):
    documentation.append(pdf_file)

print(documentation)

documents=[]

# 1. Load documents
for document in documentation:
    loader = PyPDFLoader(document)
    documents.extend(loader.load())

print(f"Loaded pages: {len(documents)}")


# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)


print(f"Loaded pages: {len(documents)}")
print(f"Created chunks: {len(docs)}")

print("Number of chunks:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(doc.page_content)


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create vector store
db = FAISS.from_documents(
    docs,
    embeddings
)

# 5. Search
query = "Can you tell me about the spottube project?"


# 7. Define structured response

class Source(BaseModel):
    file: str = Field(description="Name of the source PDF")
    page: int = Field(description="Page number in the PDF")


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question based only on retrieved context"
    )
    sources: List[Source] = Field(
        description="PDF documents and pages used to answer the question"
    )


@tool
def search_documents(query: str) -> str:
    """Search the PDF documents for information relevant to the user's query."""

    results = db.similarity_search(query, k=5)

    retrieved_documents = []

    for doc in results:

        retrieved_documents.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page", 0) + 1
        })

    return json.dumps(retrieved_documents)

# 6. Display the most relevant chunk
# context = results[0].page_content

#creating agent
agent = create_agent(
    model=chat_model,
    tools=[search_documents],
    # response_format=RAGResponse,
)


messages = [
    SystemMessage(content="""You are a helpful assistant.

Use the search_documents tool whenever the user's question
may be answered using information from the documents.

Answer the user's question using the retrieved document
content. Do not invent information that is not present
in the retrieved documents.
"""),

    HumanMessage(content=query),
]

response = agent.invoke(
    {"messages": messages},
    )

answer = response["messages"][-1].content


sources = []

for message in response["messages"]:
    if isinstance(message, ToolMessage):

        retrieved_documents = json.loads(message.content)

        for doc in retrieved_documents:
            sources.append({
                "file": doc["source"],
                "page": doc["page"]
            })

unique_sources = []

seen = set()

for source in sources:
    key = (source["file"], source["page"])

    if key not in seen:
        seen.add(key)
        unique_sources.append(source)

print("\n================ ANSWER ================\n")
print(answer)

print("\n================ SOURCES ================\n")

for source in unique_sources:
    print(f"- {source['file']} — Page {source['page']}")

# structured_response = response["structured_response"]

# print("\n================ ANSWER ================\n")
# print(structured_response.answer)

#  print("\n================ SOURCES ================\n")

# for source in structured_response.sources:
#     print(f"- {source.file} — Page {source.page}")
