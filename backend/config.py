from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = Path("documents")
UPLOAD_DIR.mkdir(exist_ok=True)

INDEX_PATH = "faiss_index"
CHECKPOINT_DB = "chat_history.sqlite"