from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import os
import fitz
from openai import OpenAI

app = FastAPI()

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# REQUEST
# =========================
class Req(BaseModel):
    question: str

# =========================
# VECTOR STORE
# =========================
store = {
    "texts": [],
    "vectors": []
}

# =========================
# EMBEDDING (LIGHT WEIGHT)
# =========================
def embed(text: str):
    v = np.zeros(128)
    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)
    return v / (np.linalg.norm(v) + 1e-8)

# =========================
# PDF EXTRACT
# =========================
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# SEARCH
# =========================
def search(qvec):
    if len(store["vectors"]) == 0:
        return []

    scores = []
    for i, v in enumerate(store["vectors"]):
        score = np.dot(qvec, v)
        scores.append((score, i))

    scores.sort(reverse=True)

    return [store["texts"][i] for _, i in scores[:5]]

# =========================
# HEALTH
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# UPLOAD (SAFE + STABLE)
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"

        contents = await file.read()

        with open(path, "wb") as f:
            f.write(contents)

        text = extract_text(path)

        if not text.strip():
            return {"error": "Cannot extract text from PDF"}

        chunks = chunk_text(text)

        # 🔥 LIMIT MEMORY (IMPORTANT FOR RENDER)
        chunks = chunks[:20]

        vectors = [embed(c) for c in chunks]

        store["texts"].extend(chunks)
        store["vectors"].extend(vectors)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "profile": {
                "document_type": "Document",
                "word_count": len(text.split()),
                "reading_time_minutes": max(1, len(text.split()) // 200)
            }
        }

    except Exception as e:
        return {"error": str(e)}

# =========================
# CHAT (REAL CHATGPT + RAG)
# =========================
@app.post("/chat")
def chat(req: Req):

    qvec = embed(req.question)
    docs = search(qvec)

    context = "\n".join(docs)

    system_prompt = """
You are a professional AI assistant.

Rules:
- Answer naturally like ChatGPT
- Use document context if relevant
- If context is not relevant, use general knowledge
- Do NOT hardcode "this is resume"
- Be helpful, clear, and concise
"""

    user_prompt = f"""
User question: {req.question}

Document context:
{context if context else "No relevant document found."}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": docs
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": docs
        }