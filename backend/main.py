from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.pdf_loader import extract_text_from_pdf
from backend.rag import SimpleRAG

app = FastAPI()

rag = SimpleRAG()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"status": "ok"}  # ❗ สำคัญมาก Render จะ detect port


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(path)

    chunks = [text[i:i+400] for i in range(0, len(text), 400)]

    rag.build(chunks)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "profile": {
            "word_count": len(text.split()),
            "document_type": "PDF"
        }
    }


@app.post("/chat")
def chat(req: dict):

    q = req["question"]

    docs = rag.search(q) if rag.index is not None else []

    context = "\n".join(docs)

    return {
        "answer": f"Based on document:\n{context[:1200]}\n\nAnswer: {q}",
        "sources": docs
    }