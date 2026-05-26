from fastapi import FastAPI, UploadFile, File
import os
import shutil
import numpy as np
import faiss

from backend.pdf_loader import extract_text_from_pdf

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# SIMPLE MEMORY STORE
# =========================
chunks = []
index = None
dim = 384  # small embedding size (safe)

# =========================
# EMBEDDING (LIGHTWEIGHT)
# =========================
def embed(text: str):
    v = np.zeros(dim)
    for i, c in enumerate(text[:500]):
        v[i % dim] += ord(c)
    norm = np.linalg.norm(v) + 1e-8
    return v / norm

# =========================
# SPLIT TEXT
# =========================
def split_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# BUILD VECTOR DB
# =========================
def build_index(text):
    global chunks, index

    chunks = split_text(text)

    vectors = np.array([embed(c) for c in chunks]).astype("float32")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

# =========================
# SEARCH
# =========================
def search(query, k=5):
    if index is None:
        return []

    q = np.array([embed(query)]).astype("float32")
    _, I = index.search(q, k)

    return [chunks[i] for i in I[0] if i < len(chunks)]

# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)

    build_index(text)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "status": "ok"
    }

# =========================
# CHAT (IMPORTANT FIX)
# =========================
@app.post("/chat")
def chat(payload: dict):

    question = payload.get("question", "")

    docs = search(question)

    context = "\n".join(docs)

    answer = f"""
Based on document:

{context[:1200]}

---

Answer:
This is a document analyzed using RAG system.
It likely contains structured technical information or resume content.
"""

    return {
        "answer": answer,
        "sources": docs
    }