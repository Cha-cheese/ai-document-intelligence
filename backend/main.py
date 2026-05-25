from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import json
import uuid

app = FastAPI()

# =========================
# STORE FILE PER SESSION
# =========================
SESSION_STORE = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# UPLOAD (RETURN SESSION ID)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        content = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(content)

        from backend.pdf_loader import extract_text_from_pdf
        from rag.chunking import chunk_text
        from rag.embeddings import get_embedding

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        vectors = [get_embedding(c) for c in chunks]

        session_id = str(uuid.uuid4())

        SESSION_STORE[session_id] = {
            "texts": chunks,
            "vectors": vectors
        }

        return {
            "ok": True,
            "session_id": session_id,
            "filename": file.filename,
            "profile": {
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 200),
                "document_type": "PDF"
            }
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# CHAT (USE SESSION ID)
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):

    try:
        import numpy as np
        from rag.embeddings import get_embedding
        from rag.llm import ask_llm

        session_id = req.session_id

        if not session_id or session_id not in SESSION_STORE:
            return {
                "answer": "Please upload a document first.",
                "sources": []
            }

        store = SESSION_STORE[session_id]

        q_vec = np.array(get_embedding(req.question))

        scores = []

        for i, v in enumerate(store["vectors"]):
            v = np.array(v)
            scores.append((np.dot(q_vec, v), i))

        scores.sort(reverse=True)

        docs = [store["texts"][i] for _, i in scores[:5]]

        context = "\n".join(docs)

        answer = ask_llm(req.question, context)

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:
        return {
            "answer": "error",
            "error": str(e),
            "sources": []
        }