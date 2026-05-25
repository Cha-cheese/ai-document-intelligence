from fastapi import FastAPI, UploadFile, File
import uuid
import os
import numpy as np
from rag.pdf_loader import load_pdf_text

app = FastAPI()

SESSION_STORE = {}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # 🔥 REAL PDF TEXT EXTRACTION
    text = load_pdf_text(file_path)

    if not text.strip():
        return {"error": "Cannot extract text from PDF"}

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    vectors = [np.random.rand(128) for _ in chunks]

    session_id = str(uuid.uuid4())

    SESSION_STORE[session_id] = {
        "chunks": chunks,
        "vectors": vectors
    }

    return {
        "ok": True,
        "session_id": session_id,
        "profile": {
            "word_count": len(text.split()),
            "document_type": "PDF"
        }
    }