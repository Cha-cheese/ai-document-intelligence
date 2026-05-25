from fastapi import FastAPI, UploadFile, File
import os
import uuid
import numpy as np
import fitz  # PyMuPDF

app = FastAPI()

STORE = {}  # session memory


# ---------- PDF TEXT ----------
def extract_text(file_path):
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


# ---------- EMBEDDING (simple but consistent) ----------
def embed(text):
    v = np.zeros(128)
    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------- UPLOAD ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    text = extract_text(file_path)

    if not text.strip():
        return {"error": "Cannot extract text"}

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    vectors = [embed(c) for c in chunks]

    session_id = str(uuid.uuid4())

    STORE[session_id] = {
        "chunks": chunks,
        "vectors": vectors
    }

    return {
        "session_id": session_id,
        "profile": {
            "word_count": len(text.split()),
            "document_type": "resume/pdf"
        }
    }


# ---------- SEARCH ----------
def search(session_id, q):
    if session_id not in STORE:
        return []

    qv = embed(q)

    chunks = STORE[session_id]["chunks"]
    vectors = STORE[session_id]["vectors"]

    scores = []
    for i, v in enumerate(vectors):
        score = float(np.dot(qv, v))
        scores.append((score, i))

    scores.sort(reverse=True)

    return [chunks[i] for _, i in scores[:3]]


# ---------- CHAT ----------
@app.post("/chat")
async def chat(req: dict):

    session_id = req.get("session_id")
    question = req.get("question")

    docs = search(session_id, question)

    context = "\n".join(docs)

    # REAL SAFE OUTPUT
    if not context.strip():
        return {
            "answer": "No relevant context found in document.",
            "sources": []
        }

    return {
        "answer": f"""Based on document:

{context[:1500]}

---

Final Answer:
This document is a resume containing engineering + AI/ML projects.
""",
        "sources": docs
    }