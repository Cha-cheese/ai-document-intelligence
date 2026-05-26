from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
import numpy as np

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
# MEMORY STORE
# =========================
DOCUMENT_TEXT = ""


# =========================
# EMBEDDING MOCK
# =========================
def embed(text):
    v = np.zeros(128)

    for i, c in enumerate(text[:1000]):
        v[i % 128] += ord(c)

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


# =========================
# SEARCH
# =========================
def search(question):

    global DOCUMENT_TEXT

    chunks = []

    text = DOCUMENT_TEXT

    size = 800

    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    if not chunks:
        return ""

    qv = embed(question)

    scores = []

    for chunk in chunks:

        cv = embed(chunk)

        score = np.dot(qv, cv)

        scores.append((score, chunk))

    scores.sort(reverse=True)

    top_chunks = [x[1] for x in scores[:3]]

    return "\n".join(top_chunks)


# =========================
# CHAT REQUEST
# =========================
class ChatRequest(BaseModel):
    question: str


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Backend running"}


# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    global DOCUMENT_TEXT

    pdf_bytes = await file.read()

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()

    DOCUMENT_TEXT = text

    return {
        "success": True,
        "filename": file.filename,
        "word_count": len(text.split())
    }


# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):

    global DOCUMENT_TEXT

    if not DOCUMENT_TEXT.strip():

        return {
            "answer": "Please upload a document first."
        }

    context = search(req.question)

    answer = f"""
Based on the uploaded document:

{context[:1500]}

Answer:
{req.question}
"""

    return {
        "answer": answer
    }