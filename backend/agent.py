import sqlite3
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent

from config import CHECKPOINT_DB
from tools import search_documents

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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

For Your information each result is separated by a "--- Source: ... ---" marker;
treat each as an independent excerpt, don't merge facts across markers unless they're clearly about the same thing.

If a search result already answers the question, answer directly from it — do not call the tool
again just to look for additional angles or sub-topics the user didn't ask about.
Only re-search if the results are clearly missing or irrelevant to what was asked.
When you answer, use all relevant information retrieved so far, not just the most recent search.
"""