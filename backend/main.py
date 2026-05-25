from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

import os
import traceback

app = FastAPI()

# SAFE IMPORT (ไม่ให้ crash)
try:
    from backend.pdf_loader import extract_text_from_pdf
    from rag.chunking import chunk_text
    from rag.embeddings import get_embedding
    from rag.llm import ask_llm
    from rag.vector_store import VectorStore
except Exception as e:
    print("IMPORT ERROR:", e)


vector_store = VectorStore() if "VectorStore" in globals() else None


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/")
def root():
    return {"status": "ok"}