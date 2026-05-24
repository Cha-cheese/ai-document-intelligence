from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.pdf_loader import extract_text_from_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.llm import ask_llm, stream_llm
from rag.vector_store import VectorStore

import json
import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI()

# ✅ IMPORTANT: keep lightweight only
vector_store = VectorStore()


class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# UPLOAD
# =========================
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

        # ✅ BATCH to reduce RAM spike
        embeddings = []
        batch_size = 4

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings.extend(get_embedding(batch))

            gc.collect()

        vector_store.add(embeddings, chunks)

        gc.collect()

        return {
            "message": "Indexed",
            "chunks": len(chunks),
            "filename": file.filename,
            "profile": {
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 220)
            }
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# CHAT (NORMAL)
# =========================
@app.post("/chat")
async def chat(request: QuestionRequest):
    try:
        query_embedding = get_embedding([request.question])[0]

        docs = vector_store.search(query_embedding, top_k=5)

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
        return {"error": str(e), "answer": "failed", "sources": []}


# =========================
# STREAM
# =========================
@app.post("/chat/stream")
async def chat_stream(request: QuestionRequest):
    try:
        query_embedding = get_embedding([request.question])[0]
        docs = vector_store.search(query_embedding, top_k=5)

        context = "\n\n".join([d["content"] for d in docs])

        def event_stream():
            yield f"event: sources\ndata: {json.dumps(docs)}\n\n"

            for token in stream_llm(
                request.question,
                context,
                history=request.history,
                mode=request.mode
            ):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            yield f"event: done\ndata: {{\"ok\": true}}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        def err():
            yield f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n"

        return StreamingResponse(err(), media_type="text/event-stream")