from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import os, json, gc
import fitz  # pymupdf
import numpy as np

app = FastAPI()

# =========================
# SIMPLE VECTOR STORE
# =========================
class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)

    def search(self, q, top_k=5):
        if not self.vectors:
            return []

        q = np.array(q)
        scores = []

        for i, v in enumerate(self.vectors):
            v = np.array(v)
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8)
            scores.append((sim, i))

        scores.sort(reverse=True)

        return [
            {"content": self.texts[i], "score": float(s)}
            for s, i in scores[:top_k]
        ]


vector_store = VectorStore()

# =========================
# EMBEDDING (LIGHT)
# =========================
def embed(text):
    vec = np.zeros(128)
    for i, c in enumerate(text[:200]):
        vec[i % 128] += ord(c)
    return (vec / (np.linalg.norm(vec) + 1e-8)).tolist()

# =========================
# PDF
# =========================
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join([p.get_text() for p in doc])

def chunk(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# UPLOAD (FIXED)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text(path)
    chunks = chunk(text)

    vectors = [embed(c) for c in chunks]
    vector_store.add(vectors, chunks)

    gc.collect()

    # 🔥 IMPORTANT FIX: ALWAYS RETURN PROFILE
    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "profile": {
            "document_type": "Document",
            "word_count": len(text.split()),
            "reading_time_minutes": max(1, len(text.split()) // 220)
        }
    }

# =========================
# CHAT STREAM
# =========================
@app.post("/chat/stream")
def chat_stream(req: dict):

    qvec = embed(req["question"])
    docs = vector_store.search(qvec)

    context = "\n".join([d["content"] for d in docs])

    def event():
        yield f"event: sources\ndata: {json.dumps(docs)}\n\n"

        answer = f"Based on document:\n{context[:800]}\n\nQ: {req['question']}"

        for word in answer.split():
            yield f"event: token\ndata: {json.dumps(word + ' ')}\n\n"

        yield f"event: done\ndata: {{\"ok\": true}}\n\n"

    return StreamingResponse(event(), media_type="text/event-stream")