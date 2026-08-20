import shutil
from pathlib import Path
from typing import List

from fastapi import UploadFile
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from config import UPLOAD_DIR


def save_uploaded_files(files: List[UploadFile]) -> List[Path]:
    saved_paths = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(dest)
    return saved_paths


def process_pdfs(paths: List[Path]) -> List[Document]:
    new_chunks = []
    for path in paths:
        filename = str(path)
        chunks = partition_pdf(
            filename=filename,
            strategy="fast",
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1200,
            new_after_n_chars=1000,
            combine_text_under_n_chars=200,
        )
        for chunk in chunks:
            new_chunks.append(
                Document(
                    page_content=str(chunk),
                    metadata={"source": filename, "type": getattr(chunk, "category", "Text")},
                )
            )
    return new_chunks