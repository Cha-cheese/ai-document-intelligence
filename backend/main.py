from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from backend.pdf_loader import load_pdf
from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.vector_store import VectorStore
from rag.llm import ask_llm

app = FastAPI()

store = VectorStore()

# =========================
# UPLOAD
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    content = await file.read()
    text = load_pdf(content)

    chunks = chunk_text(text)
    vectors = get_embedding(chunks)

    store.add(vectors, chunks)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "profile": {
            "word_count": len(text.split()),
            "document_type": "PDF"
        },
        "session_id": "default"
    }


# =========================
# CHAT
# =========================
class ChatReq(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatReq):

    q_vec = get_embedding(req.question)[0]
    docs = store.search(q_vec)

    context = "\n".join(docs)

    answer = ask_llm(req.question, context)

    return {
        "answer": answer,
        "sources": docs
    }