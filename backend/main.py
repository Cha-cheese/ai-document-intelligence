from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

import os
import traceback

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.llm import ask_llm
from rag.vector_store import VectorStore


app = FastAPI()

vector_store = VectorStore()


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


@app.get("/")
def root():
    return {"status": "ok", "message": "backend alive"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        content = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(content)

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)[:20]

        embeddings = get_embedding(chunks)

        global vector_store
        vector_store = VectorStore()
        vector_store.add(embeddings, chunks)

        return {
            "message": "ok",
            "chunks": len(chunks)
        }

    except Exception:
        return {"error": traceback.format_exc()}


@app.post("/chat")
async def chat(req: QuestionRequest):

    try:

        if len(vector_store.documents) == 0:
            return {"answer": "No document uploaded yet."}

        query_emb = get_embedding([req.question])[0]

        docs = vector_store.search(query_emb, top_k=3)

        context = "\n".join([d["content"] for d in docs])

        answer = ask_llm(req.question, context)

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception:
        return {
            "error": traceback.format_exc(),
            "answer": "failed"
        }