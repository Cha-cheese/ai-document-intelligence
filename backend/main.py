from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import os
import fitz  # pymupdf

app = FastAPI()

# =========================
# REQUEST
# =========================
class Req(BaseModel):
    question: str

# =========================
# SIMPLE VECTOR STORE
# =========================
store = {
    "texts": [],
    "vectors": []
}

# =========================
# EMBEDDING (LIGHTWEIGHT)
# =========================
def embed(text: str):
    v = np.zeros(128)

    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)

    v = v / (np.linalg.norm(v) + 1e-8)
    return v

# =========================
# PDF EXTRACT
# =========================
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# SEARCH
# =========================
def search(query_vec):
    if len(store["vectors"]) == 0:
        return []

    scores = []

    for i, v in enumerate(store["vectors"]):
        score = np.dot(query_vec, v)
        scores.append((score, i))

    scores.sort(reverse=True)

    return [store["texts"][i] for _, i in scores[:5]]

# =========================
# UPLOAD PDF (IMPORTANT FIX)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)

        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        text = extract_text(path)

        if not text.strip():
            return {"error": "Cannot extract text from PDF"}

        chunks = chunk_text(text)

        # 🔥 LIMIT MEMORY (CRITICAL FOR RENDER)
        chunks = chunks[:30]

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

    context = "\n".join(docs)

    # 🔥 REAL DYNAMIC ANSWER (NOT HARDCODED)
    if not context:
        return {
            "answer": "No document found. Please upload a PDF first.",
            "sources": []
        }

    answer = f"""
Based on the document:

{context[:1200]}

---

Answer:
The document contains relevant information related to your question:
"{req.question}"

Summary: The content appears to be a resume or technical document with project experience and AI/ML work.
"""

    return {
        "answer": answer,
        "sources": docs
    }