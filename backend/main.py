from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.vector_store import VectorStore
from rag.llm import ask_llm, stream_llm

app = FastAPI()

vector_store = VectorStore()

class QuestionRequest(BaseModel):
    question: str
    history: list = []
    mode: str = "chat"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        content = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(content)

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        vectors = []
        for c in chunks:
            vectors.append(get_embedding(c))

        vector_store.add(vectors, chunks)

        return {
            "message": "ok",
            "chunks": len(chunks),
            "filename": file.filename,
            "profile": {
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 200),
                "document_type": "PDF Document"
            }
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat(req: QuestionRequest):

    try:
        q_vec = get_embedding(req.question)

        docs = vector_store.search(q_vec, top_k=5)
        context = "\n".join(docs)

        answer = ask_llm(req.question, context)

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:
        return {
            "answer": "System error",
            "error": str(e),
            "sources": []
        }