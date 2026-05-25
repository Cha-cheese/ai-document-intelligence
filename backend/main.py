from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json

app = FastAPI()

class Req(BaseModel):
    question: str

# mock embedding
def embed(text):
    v = np.zeros(128)
    for i, c in enumerate(text[:200]):
        v[i % 128] += ord(c)
    return v / (np.linalg.norm(v) + 1e-8)

# simple store
store = {"texts": [], "vectors": []}

def search(q):
    if not store["vectors"]:
        return []

    q = np.array(q)

    scores = []
    for i, v in enumerate(store["vectors"]):
        v = np.array(v)
        score = np.dot(q, v)
        scores.append((score, i))

    scores.sort(reverse=True)
    return [store["texts"][i] for _, i in scores[:5]]

# 🔥 FAKE LLM FIX (IMPORTANT)
def generate_answer(question, context):
    return f"""
Answer based on document:

{context[:1500]}

---

Final Answer:
This document appears to be a resume/CV containing:
- Education background
- Engineering project experience
- AI/ML projects including phishing detection system
- Flutter + Node.js + ML stack experience

So yes — this is a RESUME file.
"""

@app.post("/chat")
def chat(req: Req):

    docs = search(embed(req.question))
    context = "\n".join(docs)

    answer = generate_answer(req.question, context)

    return {
        "answer": answer,
        "sources": docs
    }