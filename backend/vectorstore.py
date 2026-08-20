import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from config import INDEX_PATH

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = None


def load_existing_index():
    global db
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("No FAISS index found yet — starting empty. Upload PDFs to build one.")
    return db


def add_documents(chunks):
    global db
    if not chunks:
        return
    if db is None:
        db = FAISS.from_documents(chunks, embeddings)
    else:
        db.add_documents(chunks)
    db.save_local(INDEX_PATH)


def get_db():
    return db


load_existing_index()