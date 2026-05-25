from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

app = FastAPI()

# =========================
# GLOBAL MEMORY (ต้องอยู่ระดับนี้เท่านั้น)
# =========================
VECTOR_STORE = {
    "texts": [],
    "vectors": []
}


# =========================
# MODELS
# =========================
class QuestionRequest(BaseModel):
    question: str


# =========================
# ROOT DEBUG
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# UPLOAD
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

        # 🔥 IMPORTANT FIX: overwrite global memory
        VECTOR_STORE["texts"] = chunks
        VECTOR_STORE["vectors"] = vectors

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
# CHAT
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):

    try:
        from rag.embeddings import get_embedding
        from rag.llm import ask_llm
        import numpy as np

        if not VECTOR_STORE["vectors"]:
            return {
                "answer": "Please upload a document first.",
                "sources": []
            }

        q_vec = np.array(get_embedding(req.question))

        scores = []
        for i, v in enumerate(VECTOR_STORE["vectors"]):
            v = np.array(v)
            score = np.dot(q_vec, v)
            scores.append((score, i))

        scores.sort(reverse=True)

        docs = [VECTOR_STORE["texts"][i] for _, i in scores[:5]]

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