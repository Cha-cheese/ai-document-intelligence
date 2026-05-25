from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.llm import ask_llm
from rag.vector_store import VectorStore

import os
import traceback

app = FastAPI()

vector_store = VectorStore()


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)

        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(path)

        chunks = chunk_text(text)

        if len(chunks) > 30:
            chunks = chunks[:30]

        embeddings = get_embedding(chunks)

        global vector_store
        vector_store = VectorStore()

        vector_store.add(embeddings, chunks)

        return {
            "message": "Indexed successfully",
            "filename": file.filename,
            "chunks": len(chunks),
            "profile": {
                "document_type": "PDF Document"
            }
        }

    except Exception:

        return {
            "error": traceback.format_exc()
        }


@app.post("/chat")
async def chat(request: QuestionRequest):

    try:

        if len(vector_store.documents) == 0:

            return {
                "answer": "Please upload a document first.",
                "sources": []
            }

        query_embedding = get_embedding(request.question)[0]

        docs = vector_store.search(
            query_embedding,
            top_k=5
        )

        context = "\n\n".join([
            d["content"]
            for d in docs
        ])

        answer = ask_llm(
            request.question,
            context
        )

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception:

        return {
            "error": traceback.format_exc(),
            "answer": "AI processing error occurred.",
            "sources": []
        }