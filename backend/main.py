from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import os, json, gc, re
import numpy as np
import fitz  # pymupdf

# =========================
# FASTAPI
# =========================
app = FastAPI()

# =========================
# MEMORY SAFE VECTOR STORE (IN-MEMORY)
# =========================
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)

    def search(self, query_vec, top_k=5):
        if not self.vectors:
            return []

        sims = []
        q = np.array(query_vec)

        for i, v in enumerate(self.vectors):
            v = np.array(v)
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8)
            sims.append((sim, i))

        sims.sort(reverse=True)
        results = []

        for score, i in sims[:top_k]:
            results.append({
                "content": self.texts[i],
                "score": float(score)
            })

        return results


vector_store = SimpleVectorStore()

# =========================
# MODEL
# =========================
class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)

# =========================
# EMBEDDING (NO AI API → SAFE LOCAL HASH)
# =========================
def embed(text: str):
    # ultra lightweight embedding (no torch, no HF, no OpenAI)
    vec = np.zeros(128)
    for i, c in enumerate(text[:200]):
        vec[i % 128] += ord(c)
    return (vec / (np.linalg.norm(vec) + 1e-8)).tolist()

# =========================
# PDF
# =========================
def extract_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# SIMPLE LLM (NO API DEPENDENCY)
# =========================
def fake_llm(question, context):
    return f"""Answer based on document:

{context[:800]}

---
Q: {question}

This is a lightweight response (no external API)."""

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# UPLOAD
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)

    text = extract_pdf(path)
    chunks = chunk(text)

    vectors = [embed(c) for c in chunks]
    vector_store.add(vectors, chunks)

    gc.collect()

    return {
        "message": "ok",
        "chunks": len(chunks),
        "filename": file.filename
    }

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):
    qvec = embed(req.question)
    docs = vector_store.search(qvec)

    context = "\n".join([d["content"] for d in docs])

    answer = fake_llm(req.question, context)

    return {
        "answer": answer,
        "sources": docs
    }

# =========================
# STREAM
# =========================
@app.post("/chat/stream")
def stream(req: QuestionRequest):

    qvec = embed(req.question)
    docs = vector_store.search(qvec)
    context = "\n".join([d["content"] for d in docs])

    def event():
        yield f"event: sources\ndata: {json.dumps(docs)}\n\n"

        for word in fake_llm(req.question, context).split():
            yield f"event: token\ndata: {json.dumps(word + ' ')}\n\n"

        yield f"event: done\ndata: {{\"ok\": true}}\n\n"

    return StreamingResponse(event(), media_type="text/event-stream")