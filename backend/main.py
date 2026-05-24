from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import os
import json
import gc

# =====================
# SAFE IMPORT (lazy later)
# =====================

app = FastAPI()

vector_store = None


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


@app.get("/")
def root():
    return {"status": "ok"}


# =====================
# LAZY LOAD FUNCTIONS
# =====================

def get_vector_store():
    global vector_store
    if vector_store is None:
        from rag.vector_store import VectorStore
        vector_store = VectorStore()
    return vector_store


def get_embedding():
    from rag.embeddings import get_embedding as emb
    return emb


def get_llm():
    from rag.llm import ask_llm, stream_llm
    return ask_llm, stream_llm


def get_pdf_loader():
    from backend.pdf_loader import extract_text_from_pdf
    return extract_text_from_pdf


def get_chunker():
    from rag.chunking import chunk_text
    return chunk_text


# =====================
# UPLOAD
# =====================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        extract_text_from_pdf = get_pdf_loader()
        chunk_text = get_chunker()
        VectorStore = get_vector_store()

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        embeddings_fn = get_embedding()

        embeddings = []
        for i in range(0, len(chunks), 4):
            batch = chunks[i:i+4]
            embeddings.extend(embeddings_fn(batch))
            gc.collect()

        store = get_vector_store()
        store.add(embeddings, chunks)

        gc.collect()

        return {
            "message": "Indexed",
            "chunks": len(chunks),
            "filename": file.filename
        }

    except Exception as e:
        return {"error": str(e)}


# =====================
# CHAT
# =====================
@app.post("/chat")
async def chat(request: QuestionRequest):

    try:
        embeddings_fn = get_embedding()
        ask_llm, _ = get_llm()

        store = get_vector_store()

        query_emb = embeddings_fn([request.question])[0]
        docs = store.search(query_emb, top_k=5)

        context = "\n\n".join([d["content"] for d in docs])

        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:
        return {"error": str(e)}