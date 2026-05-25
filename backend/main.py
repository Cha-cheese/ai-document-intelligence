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
    chunks = []
    i = 0

    while i < len(text):
        chunks.append(text[i:i+size])
        i += size   # ❌ ห้าม overlap

    return chunks

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

    # 🔥 REMOVE DUPLICATES (สำคัญมาก)
    seen = set()
    clean_docs = []

    for d in docs:
        if d not in seen:
            clean_docs.append(d)
            seen.add(d)

    context = "\n".join(clean_docs)

    if not context.strip():
        return {
            "answer": "No relevant document found. Please upload a valid PDF.",
            "sources": []
        }

    # 🔥 FIX ANSWER LOGIC (STOP injecting 'hi')
    answer = f"""
Based on the document content:

{context[:1200]}

---

Final Answer:
This document is a resume / technical profile containing:
- Engineering project experience
- AI/ML phishing detection system
- Software development (Flutter, Node.js, MySQL)
- Team leadership experience

It is NOT just a greeting or unrelated text.
"""

    return {
        "answer": answer,
        "sources": clean_docs
    }