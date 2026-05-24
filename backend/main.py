from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import os
import json
import gc

app = FastAPI()

# =========================
# LAZY GLOBALS (สำคัญมาก)
# =========================
vector_store = None


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
# LAZY IMPORT HELPERS
# =========================
def get_vector_store():
    global vector_store
    if vector_store is None:
        from rag.vector_store import VectorStore
        vector_store = VectorStore()
    return vector_store


def get_embedding():
    from rag.embeddings import get_embedding as emb
    return emb


def get_llm():
    from rag.llm import ask_llm
    return ask_llm


def get_pdf():
    from backend.pdf_loader import extract_text_from_pdf
    return extract_text_from_pdf


def get_chunk():
    from rag.chunking import chunk_text
    return chunk_text


# =========================
# UPLOAD (SAFE)
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        extract_text_from_pdf = get_pdf()
        chunk_text = get_chunk()
        embeddings_fn = get_embedding()

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        embeddings = []
        batch_size = 3  # 🔥 small batch to prevent OOM

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings.extend(embeddings_fn(batch))
            gc.collect()

        store = get_vector_store()
        store.add(embeddings, chunks)

        gc.collect()

        return {
            "message": "Indexed successfully",
            "chunks": len(chunks),
            "filename": file.filename,
            "profile": {
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 220)
            }
        }

    except Exception as e:
        return {"error": str(e), "message": "upload failed"}


# =========================
# CHAT (NO STREAM = STABLE)
# =========================
@app.post("/chat")
async def chat(request: QuestionRequest):
    try:
        embeddings_fn = get_embedding()
        ask_llm = get_llm()
        store = get_vector_store()

        query_emb = embeddings_fn([request.question])[0]
        docs = store.search(query_emb, top_k=5)

        context = "\n\n".join([d["content"] for d in docs])

        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:
        return {
            "error": str(e),
            "answer": "AI failed",
            "sources": []
        }


# =========================
# STREAM (DISABLED SAFE VERSION)
# =========================
@app.post("/chat/stream")
async def chat_stream(request: QuestionRequest):
    """
    ⚠️ Safe fallback streaming (NOT real token streaming)
    prevents Render 502 crash
    """
    try:
        embeddings_fn = get_embedding()
        store = get_vector_store()
        ask_llm = get_llm()

        query_emb = embeddings_fn([request.question])[0]
        docs = store.search(query_emb, top_k=5)

        context = "\n\n".join([d["content"] for d in docs])

        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        def stream():
            yield f"event: sources\ndata: {json.dumps(docs)}\n\n"
            yield f"event: token\ndata: {json.dumps(answer)}\n\n"
            yield f"event: done\ndata: {{\"ok\": true}}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    except Exception as e:

        def err():
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(err(), media_type="text/event-stream")