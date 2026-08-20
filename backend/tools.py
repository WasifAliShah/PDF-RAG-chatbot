import json
from langchain_core.tools import tool

import vectorstore


@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF documents for information relevant to the user's query."""
    db = vectorstore.get_db()
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