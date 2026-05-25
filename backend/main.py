from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import os
import numpy as np

app = FastAPI()

SESSION_STORE = {}


class ChatReq(BaseModel):
    question: str
    session_id: str


@app.get("/")
def home():
    return {"status": "ok"}


# =========================
# UPLOAD
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    content = await file.read()

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(content)

    # mock extract
    text = content.decode("utf-8", errors="ignore")

    chunks = [text[i:i+300] for i in range(0, len(text), 300)]

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


# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatReq):

    if req.session_id not in SESSION_STORE:
        return {"answer": "Please upload a document first."}

    store = SESSION_STORE[req.session_id]

    q_vec = np.random.rand(128)

    scores = [
        (np.dot(q_vec, v), i)
        for i, v in enumerate(store["vectors"])
    ]

    scores.sort(reverse=True)

    context = "\n".join(
        store["chunks"][i] for _, i in scores[:3]
    )

    return {
        "answer": f"Answer based on document:\n\n{context[:800]}",
        "sources": store["chunks"][:3]
    }