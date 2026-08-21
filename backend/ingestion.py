import shutil
from pathlib import Path
from typing import List

from fastapi import UploadFile
# from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
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


SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def process_documents(paths: List[Path]) -> List[Document]:
    new_chunks = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        filename = str(path)
        kwargs = dict(
            filename=filename,
            chunking_strategy="by_title",
            max_characters=1200,
            new_after_n_chars=1000,
            combine_text_under_n_chars=200,
            infer_table_structure=True,
        )
        if path.suffix.lower() == ".pdf":
            kwargs["strategy"] = "hi_res"  # only meaningful for PDF/image; leave unset for docx

        elements = partition(**kwargs)
        for el in elements:
            new_chunks.append(Document(
                page_content=str(el),
                metadata={
                    "source": filename,
                    "type": getattr(el, "category", "Text"),
                    "page": getattr(el.metadata, "page_number", None),
                },
            ))
    return new_chunks