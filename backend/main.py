import os
from fastapi import FastAPI, UploadFile, File
import shutil

from backend.pdf_loader import extract_text_from_pdf
from backend.rag import SimpleRAG

app = FastAPI()

rag = SimpleRAG()
DOC_TEXT = []

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------
# UPLOAD PDF
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)

    # chunking simple
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    rag.build(chunks)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "profile": {
            "word_count": len(text.split()),
            "document_type": "PDF"
        }
    }


# --------------------
# CHAT (RAG)
# --------------------
@app.post("/chat")
async def chat(req: dict):

    question = req["question"]

    docs = rag.search(question)

    context = "\n".join(docs)

    answer = f"""
Based on document:

{context[:1500]}

---

Answer:
{question} relates to the document content above.
"""

    return {
        "answer": answer,
        "sources": docs
    }