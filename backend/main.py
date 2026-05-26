from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import fitz
import numpy as np

from rag.embeddings import get_embedding
from rag.vector_store import (
    add_documents,
    search
)

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL STATE
# =========================

document_uploaded = False


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    question: str


# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_text_from_pdf(file_bytes):

    text = ""

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text()

    return text


# =========================
# TEXT CHUNKING
# =========================

def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =========================
# HEALTH
# =========================

@app.get("/")
def health():

    return {
        "status": "running"
    }


# =========================
# UPLOAD
# =========================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global document_uploaded

    content = await file.read()

    text = extract_text_from_pdf(content)

    chunks = chunk_text(text)

    embeddings = []

    for chunk in chunks:

        embedding = get_embedding(chunk)

        embeddings.append(embedding)

    add_documents(chunks, embeddings)

    document_uploaded = True

    return {
        "success": True,
        "filename": file.filename,
        "chunks": len(chunks),
        "profile": {
            "word_count": len(text.split()),
            "document_type": "PDF Document",
            "reading_time_minutes": max(
                1,
                len(text.split()) // 200
            )
        }
    }


# =========================
# CHAT
# =========================

@app.post("/chat")
def chat(req: ChatRequest):

    global document_uploaded

    if not document_uploaded:

        return {
            "answer": "Please upload a document first."
        }

    query_embedding = get_embedding(
        req.question
    )

    docs = search(query_embedding)

    context = "\n\n".join(docs)

    answer = f"""
Based on the uploaded document:

{context[:3000]}

Question:
{req.question}

Answer:
This answer was generated using the uploaded document context.
"""

    return {
        "answer": answer
    }