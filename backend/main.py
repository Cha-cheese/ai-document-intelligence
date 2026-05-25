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

gc.collect()

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

    global vector_store

    try:

        contents = await file.read()

        os.makedirs("uploads", exist_ok=True)

        upload_path = f"uploads/{file.filename}"

        with open(upload_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(upload_path)

        chunks = chunk_text(text)

        embeddings = []

        batch_size = 20

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            batch_embeddings = get_embedding(batch)

            embeddings.extend(batch_embeddings)

        vector_store = VectorStore()

        vector_store.add(embeddings, chunks)

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

    except Exception as e:

        return {
            "error": str(e),
            "message": "Upload failed"
        }


@app.post("/chat")
async def chat(request: QuestionRequest):

    try:

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

    except Exception as e:

        return {
            "error": str(e),
            "answer": "AI processing error occurred.",
            "sources": []
        }


@app.post("/chat/stream")
async def chat_stream(request: QuestionRequest):

    try:

        query_embedding = get_embedding(
        [request.question]
        )[0]

        retrieved_docs = vector_store.search(
            query_embedding,
            top_k=3
        )

        context = "\n\n".join([
            doc["content"]
            for doc in retrieved_docs
        ])

        def event_stream():

            yield (
                f"event: sources\n"
                f"data: {json.dumps(retrieved_docs)}\n\n"
            )

            for token in stream_llm(
                request.question,
                context,
                history=request.history,
                mode=request.mode
            ):

                yield (
                    f"event: token\n"
                    f"data: {json.dumps(token)}\n\n"
                )

            yield (
                f"event: done\n"
                f"data: {{\"status\":\"complete\"}}\n\n"
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:

        def error_stream():

            yield (
                f"event: error\n"
                f"data: {json.dumps({'message': str(e)})}\n\n"
            )

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )