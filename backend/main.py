from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from backend.pdf_loader import extract_text_from_pdf
from backend.vector_store import VectorStore
from rag.embedding import embed

app = FastAPI()

store = VectorStore()


# -------------------------
# MODELS
# -------------------------
class ChatReq(BaseModel):
    question: str


# -------------------------
# UPLOAD PDF
# -------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    content = await file.read()
    text = extract_text_from_pdf(content)

    # simple chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    for c in chunks:
        vec = embed(c)
        store.add(c, vec)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "status": "ok"
    }


# -------------------------
# CHAT (NO STREAM = STABLE)
# -------------------------
@app.post("/chat")
def chat(req: ChatReq):

    q_vec = embed(req.question)
    docs = store.search(q_vec)

    context = "\n".join(docs)

    answer = f"""
Based on your document:

{context[:1500]}

---

Answer:
This document is a resume / technical profile with AI + software engineering projects.
"""

    return {
        "answer": answer,
        "sources": docs
    }


# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"status": "running"}