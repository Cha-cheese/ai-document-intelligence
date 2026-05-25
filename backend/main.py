from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.llm import ask_llm
from rag.vector_store import VectorStore

import os
import gc
import json
import traceback

# =========================
# ENV
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

gc.collect()

# =========================
# APP
# =========================
app = FastAPI()

DATA_FILE = "vector_store.json"


# =========================
# REQUEST MODEL
# =========================
class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# SAVE VECTOR STORE
# =========================
def save_vector_store(embeddings, chunks):

    data = []

    for embedding, chunk in zip(
        embeddings,
        chunks
    ):

        data.append({
            "embedding": embedding,
            "content": chunk
        })

    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


# =========================
# LOAD VECTOR STORE
# =========================
def load_vector_store():

    vector_store = VectorStore()

    if not os.path.exists(DATA_FILE):
        return vector_store

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    embeddings = []
    documents = []

    for item in data:

        embeddings.append(item["embedding"])
        documents.append(item["content"])

    vector_store.add(
        embeddings,
        documents
    )

    return vector_store


# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)

        upload_path = f"uploads/{file.filename}"

        with open(upload_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(upload_path)

        if not text.strip():

            return {
                "error": "No text found"
            }

        chunks = chunk_text(text)

        embeddings = []

        batch_size = 20

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            batch_embeddings = get_embedding(batch)

            embeddings.extend(batch_embeddings)

        # =========================
        # SAVE TO FILE
        # =========================
        save_vector_store(
            embeddings,
            chunks
        )

        gc.collect()

        return {
            "message": "Indexed successfully",
            "chunks": len(chunks),
            "filename": file.filename,
            "profile": {
                "document_type": "General Document",
                "word_count": len(text.split()),
                "reading_time_minutes": max(
                    1,
                    len(text.split()) // 220
                )
            }
        }

    except Exception:

        return {
            "error": traceback.format_exc()
        }


# =========================
# CHAT
# =========================
@app.post("/chat")
async def chat(request: QuestionRequest):

    try:

        # =========================
        # LOAD VECTOR STORE
        # =========================
        vector_store = load_vector_store()

        if len(vector_store.documents) == 0:

            return {
                "answer": "No document uploaded yet.",
                "sources": []
            }

        query_embedding = get_embedding(
            [request.question]
        )[0]

        retrieved_docs = vector_store.search(
            query_embedding,
            top_k=2
        )

        context = "\n\n".join([
            doc["content"]
            for doc in retrieved_docs
        ])

        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        return {
            "answer": answer,
            "sources": retrieved_docs
        }

    except Exception:

        return {
            "answer": traceback.format_exc(),
            "sources": []
        }