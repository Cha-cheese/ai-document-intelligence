from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

app = FastAPI()

# =========================
# SAFE INIT (กัน crash ตอน boot)
# =========================
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        from rag.vector_store import VectorStore
        vector_store = VectorStore()
    return vector_store


# =========================
# REQUEST
# =========================
class QuestionRequest(BaseModel):
    question: str


# =========================
# ROOT (สำคัญ debug render)
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# UPLOAD (SAFE)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        from backend.pdf_loader import extract_text_from_pdf
        from rag.chunking import chunk_text
        from rag.embeddings import get_embedding

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        vectors = [get_embedding(c) for c in chunks]

        vs = get_vector_store()
        vs.add(vectors, chunks)

        return {
            "message": "uploaded",
            "chunks": len(chunks),
            "filename": file.filename,
            "profile": {
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 200),
                "document_type": "PDF"
            }
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# CHAT (SAFE)
# =========================
@app.post("/chat")
def chat(req: QuestionRequest):

    try:
        from rag.embeddings import get_embedding
        from rag.llm import ask_llm

        vs = get_vector_store()

        q_vec = get_embedding(req.question)
        docs = vs.search(q_vec, top_k=5)

        context = "\n".join(docs)

        answer = ask_llm(req.question, context)

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:
        return {
            "answer": "system error",
            "error": str(e),
            "sources": []
        }