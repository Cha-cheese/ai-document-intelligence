from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.vector_store import VectorStore

import os
import json
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

vector_store = VectorStore()


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    contents = await file.read()

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)

    embeddings = get_embedding(chunks)

    vector_store.add(embeddings, chunks)

    gc.collect()

    return {
        "message": "ok",
        "chunks": len(chunks),
        "filename": file.filename,
        "profile": {
            "word_count": len(text.split())
        }
    }


@app.post("/chat")
async def chat(req: QuestionRequest):

    query_emb = get_embedding([req.question])[0]

    docs = vector_store.search(query_emb)

    context = "\n\n".join([d["content"] for d in docs])

    # simple fallback (ไม่ใช้ LLM เพื่อไม่พัง)
    answer = f"I found {len(docs)} relevant sections:\n\n{context[:1000]}"

    return {
        "answer": answer,
        "sources": docs
    }