from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import os
import fitz

from rag.vector_store import VectorStore
from rag.llm import ask_llm

app = FastAPI()

# =========================
# INIT VECTOR DB
# =========================
vector_store = VectorStore(dim=128)

# =========================
# REQUEST
# =========================
class QuestionRequest(BaseModel):
    question: str

# =========================
# EMBEDDING (lightweight)
# =========================
def get_embedding(text):
    v = np.zeros(128)
    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)

    return v / (np.linalg.norm(v) + 1e-8)

# =========================
# PDF LOADER
# =========================
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "RAG system running"}

# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        text = extract_text(path)
        chunks = chunk_text(text)

        # 🔥 limit for stability
        chunks = chunks[:30]

        embeddings = [get_embedding(c) for c in chunks]

        vector_store.add(embeddings, chunks)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "message": "Indexed successfully"
        }

    except Exception as e:
        return {"error": str(e)}

# =========================
# CHAT (RAG CORE)
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):

    query_vec = get_embedding(req.question)

    docs = vector_store.search(query_vec, top_k=5)

    context = "\n".join(docs)

    answer = ask_llm(
        req.question,
        context
    )

    return {
        "answer": answer,
        "sources": docs
    }