from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import json

app = FastAPI()

DATA_PATH = "data_store.json"


# =========================
# LOAD / SAVE MEMORY
# =========================
def load_store():
    if not os.path.exists(DATA_PATH):
        return {"texts": [], "vectors": []}

    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save_store(store):
    with open(DATA_PATH, "w") as f:
        json.dump(store, f)


# =========================
# REQUEST
# =========================
class QuestionRequest(BaseModel):
    question: str


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# UPLOAD (PERSISTENT FIX)
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

        vectors = [get_embedding(c).tolist() for c in chunks]

        store = {
            "texts": chunks,
            "vectors": vectors
        }

        save_store(store)

        return {
            "ok": True,
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
# CHAT (LOAD FROM DISK)
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):

    try:
        import numpy as np
        from rag.embeddings import get_embedding
        from rag.llm import ask_llm

        store = load_store()

        if not store["texts"]:
            return {
                "answer": "Please upload a document first.",
                "sources": []
            }

        q_vec = np.array(get_embedding(req.question))

        scores = []

        for i, v in enumerate(store["vectors"]):
            v = np.array(v)
            score = np.dot(q_vec, v)
            scores.append((score, i))

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