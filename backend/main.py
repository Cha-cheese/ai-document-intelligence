from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import os
import fitz

app = FastAPI()

# =========================
# REQUEST
# =========================
class Req(BaseModel):
    question: str

# =========================
# SIMPLE VECTOR STORE (SAFE)
# =========================
store = {
    "texts": [],
    "vectors": []
}

# =========================
# EMBEDDING (LIGHTWEIGHT, NO API)
# =========================
def embed(text: str):
    v = np.zeros(128)

    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)

    return v / (np.linalg.norm(v) + 1e-8)

# =========================
# PDF READER
# =========================
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# SEARCH
# =========================
def search(qvec):
    if len(store["vectors"]) == 0:
        return []

    scores = []
    for i, v in enumerate(store["vectors"]):
        score = np.dot(qvec, v)
        scores.append((score, i))

    scores.sort(reverse=True)

    return [store["texts"][i] for _, i in scores[:5]]

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# UPLOAD (STABLE VERSION)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        contents = await file.read()

        if not contents:
            return {"error": "Empty file"}

        with open(path, "wb") as f:
            f.write(contents)

        text = extract_text(path)

        if not text or len(text.strip()) == 0:
            return {"error": "Cannot extract text from PDF"}

        chunks = chunk_text(text)

        # 🔥 IMPORTANT: prevent Render OOM
        chunks = chunks[:20]

        vectors = [embed(c) for c in chunks]

        store["texts"].extend(chunks)
        store["vectors"].extend(vectors)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "profile": {
                "document_type": "Document",
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 200)
            }
        }

    except Exception as e:
        return {"error": str(e)}

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: Req):

    qvec = embed(req.question)
    docs = search(qvec)

    if not docs:
        return {
            "answer": "No document uploaded yet or no data found.",
            "sources": []
        }

    context = "\n".join(docs)

    answer = f"""
Based on the document:

{context[:1200]}

---

Answer:
The document contains relevant information related to:
{req.question}

This appears to be a resume or technical document with engineering + AI/ML experience.
"""

    return {
        "answer": answer,
        "sources": docs
    }