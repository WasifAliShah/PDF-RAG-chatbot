import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_core.tools import tool
# data class for user ids
from dataclasses import dataclass
# for context saving
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv


load_dotenv()

# 2. Initialize the Gemini model (e.g., gemini-1.5-flash)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.9)

documentation=["AI_Engineer.pdf"]

documents=[]

# 1. Load documents
for document in documentation:
    loader = PyPDFLoader(document)
    documents.extend(loader.load())

print(f"Loaded pages: {len(documents)}")


# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)


print(f"Loaded pages: {len(documents)}")
print(f"Created chunks: {len(docs)}")

print("Number of chunks:", len(docs))

# for i, doc in enumerate(docs):
#     print(f"\n========== CHUNK {i + 1} ==========")
#     print(doc.page_content)


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

@tool
def search_documents(query: str) -> str:
  """Search the company documents for information relevant to the query."""
  results = db.similarity_search(query, k=5)

  context = "\n\n".join(
      doc.page_content for doc in results
  )
  return context

# 6. Display the most relevant chunk
# context = results[0].page_content

#creating agent
agent = create_agent(
    model=llm,
    tools=[search_documents],
)


messages = [
    SystemMessage(content="""You are a helpful assistant.

Use the search_documents tool whenever the user asks
about information that may be contained in the documents."""
),
    HumanMessage(content=query),
]

response = agent.invoke(
    {"messages": messages},
    )

print(response["messages"][-1].content)
