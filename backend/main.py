from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.llm import ask_llm
from rag.vector_store import VectorStore

import os
import gc
import traceback

# =========================
# MEMORY OPTIMIZATION
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

gc.collect()

# =========================
# FASTAPI
# =========================
app = FastAPI()

# =========================
# GLOBAL VECTOR STORE
# =========================
vector_store = VectorStore()

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

    return {
        "status": "ok"
    }


# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global vector_store

    try:

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)

        upload_path = f"uploads/{file.filename}"

        with open(upload_path, "wb") as f:
            f.write(contents)

        print("PDF SAVED")

        # =========================
        # EXTRACT TEXT
        # =========================
        text = extract_text_from_pdf(upload_path)

        print("TEXT EXTRACTED")

        if not text or len(text.strip()) == 0:

            return {
                "error": "No text found in PDF",
                "message": "Upload failed"
            }

        # =========================
        # CHUNK
        # =========================
        chunks = chunk_text(text)

        print("CHUNKS:", len(chunks))

        if len(chunks) == 0:

            return {
                "error": "No chunks generated",
                "message": "Upload failed"
            }

        # =========================
        # EMBEDDINGS
        # =========================
        embeddings = []

        batch_size = 20

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            batch_embeddings = get_embedding(batch)

            embeddings.extend(batch_embeddings)

        print("EMBEDDINGS:", len(embeddings))

        # =========================
        # RESET VECTOR STORE
        # =========================
        vector_store = VectorStore()

        vector_store.add(
            embeddings,
            chunks
        )

        print("VECTOR STORE READY")

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

        error_message = traceback.format_exc()

        print(error_message)

        return {
            "error": error_message,
            "message": "Upload failed"
        }


# =========================
# CHAT
# =========================
@app.post("/chat")
async def chat(request: QuestionRequest):

    global vector_store

    try:

        print("QUESTION:", request.question)

        # =========================
        # CHECK VECTOR STORE
        # =========================
        if len(vector_store.documents) == 0:

            return {
                "answer": "Please upload a document first.",
                "sources": []
            }

        # =========================
        # QUERY EMBEDDING
        # =========================
        query_embedding = get_embedding(
            [request.question]
        )[0]

        print("QUERY EMBEDDING READY")

        # =========================
        # SEARCH
        # =========================
        retrieved_docs = vector_store.search(
            query_embedding,
            top_k=2
        )

        print("DOCS FOUND:", len(retrieved_docs))

        if len(retrieved_docs) == 0:

            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        # =========================
        # CONTEXT
        # =========================
        context = "\n\n".join([
            doc["content"]
            for doc in retrieved_docs
        ])

        print("CONTEXT READY")

        # =========================
        # LLM
        # =========================
        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        print("ANSWER:", answer)

        return {
            "answer": answer,
            "sources": retrieved_docs
        }

    except Exception:

        error_message = traceback.format_exc()

        print(error_message)

        return {
            "answer": error_message,
            "sources": []
        }