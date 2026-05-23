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
import re

app = FastAPI()

# -------------------------
# VECTOR STORE (FIXED)
# -------------------------
vector_store = VectorStore()   # ❌ removed dimension=384


# -------------------------
# REQUEST MODEL (FIXED MISSING CLASS)
# -------------------------
class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = "Analyze"


def detect_document_profile(text):
    lowered_text = text.lower()
    topic_rules = {
        "AI / Machine Learning": ["machine learning", "random forest", "tf-idf", "model", "dataset", "ai"],
        "Backend": ["backend", "node.js", "api", "server", "database"],
        "Frontend / Mobile": ["flutter", "frontend", "ui", "ux", "mobile"],
        "Security": ["phishing", "security", "threat", "email detection"],
        "Research": ["abstract", "methodology", "results", "references", "experiment"],
        "Legal / Contract": ["agreement", "contract", "party", "liability", "termination"]
    }
    topics = [
        topic
        for topic, keywords in topic_rules.items()
        if any(keyword in lowered_text for keyword in keywords)
    ]

    if any(keyword in lowered_text for keyword in ["resume", "education", "experience", "skills", "project"]):
        document_type = "Resume / Profile"
    elif any(keyword in lowered_text for keyword in ["agreement", "contract", "whereas"]):
        document_type = "Contract / Legal"
    elif any(keyword in lowered_text for keyword in ["abstract", "methodology", "references"]):
        document_type = "Research / Report"
    else:
        document_type = "General Document"

    keywords = re.findall(r"[A-Za-z][A-Za-z+#.-]{2,}", text)
    stopwords = {
        "and", "for", "the", "with", "from", "this", "that", "are", "was",
        "were", "have", "has", "using", "project", "resume", "email"
    }
    keyword_counts = {}

    for keyword in keywords:
        normalized_keyword = keyword.lower()

        if normalized_keyword in stopwords:
            continue

        keyword_counts[normalized_keyword] = keyword_counts.get(normalized_keyword, 0) + 1

    top_keywords = [
        keyword
        for keyword, _ in sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    words = re.findall(r"\w+", text)

    return {
        "document_type": document_type,
        "topics": topics[:6] or ["General"],
        "keywords": top_keywords,
        "word_count": len(words),
        "reading_time_minutes": max(1, round(len(words) / 220))
    }


def build_context(retrieved_docs):
    context_blocks = []

    for doc in retrieved_docs:
        context_blocks.append(
            f"[Source #{doc['source_id']} | Similarity: {doc['score']}]\n"
            f"{doc['content']}"
        )

    return "\n\n---\n\n".join(context_blocks)


def extract_question_terms(question):
    stopwords = {
        "about", "after", "again", "could", "does", "from", "have", "into",
        "more", "should", "that", "their", "there", "these", "this", "what",
        "when", "where", "which", "with", "would", "your"
    }
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.-]{2,}", question.lower())
    return [term for term in terms if term not in stopwords]


def explain_source(doc, question):
    terms = extract_question_terms(question)
    content = doc["content"].lower()
    matched_terms = sorted({term for term in terms if term in content})

    if matched_terms:
        evidence = f" It directly matches question terms: {', '.join(matched_terms[:6])}."
    else:
        evidence = " It was selected by semantic similarity even without exact keyword overlap."

    return (
        "This source was retrieved as supporting evidence for the question. "
        f"It has semantic similarity of {doc['score']}.{evidence}"
    )


def enrich_sources(retrieved_docs, question):
    for rank, doc in enumerate(retrieved_docs, start=1):
        terms = extract_question_terms(question)
        content = doc["content"].lower()
        matched_terms = sorted({term for term in terms if term in content})

        doc["rank"] = rank
        doc["matched_terms"] = matched_terms
        doc["explanation"] = explain_source(doc, question)

    return retrieved_docs


def sse_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Doc Assistant running"}


# -------------------------
# UPLOAD PDF
# -------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    contents = await file.read()

    os.makedirs("uploads", exist_ok=True)
    upload_path = f"uploads/{file.filename}"

    with open(upload_path, "wb") as f:
        f.write(contents)

    text = extract_text_from_pdf(upload_path)
    chunks = chunk_text(text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    vector_store = VectorStore()
    vector_store.add(embeddings, chunks)
    profile = detect_document_profile(text)

    return {
        "message": "Indexed successfully",
        "chunks": len(chunks),
        "filename": file.filename,
        "profile": profile
    }


# -------------------------
# CHAT
# -------------------------
@app.post("/chat")
async def chat(request: QuestionRequest):

    try:
        # EMBED QUESTION
        query_embedding = get_embedding(request.question)

        # SEARCH
        retrieved_docs = vector_store.search(query_embedding, top_k=5)

        # BUILD CONTEXT
        context = build_context(retrieved_docs)

        # LLM RESPONSE
        answer = ask_llm(
            request.question,
            context,
            history=request.history,
            mode=request.mode
        )

        # SOURCES
        retrieved_docs = enrich_sources(retrieved_docs, request.question)

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
        query_embedding = get_embedding(request.question)
        retrieved_docs = vector_store.search(query_embedding, top_k=5)
        context = build_context(retrieved_docs)
        retrieved_docs = enrich_sources(retrieved_docs, request.question)

        def event_stream():
            yield sse_event("sources", retrieved_docs)

            for token in stream_llm(
                request.question,
                context,
                history=request.history,
                mode=request.mode
            ):
                yield sse_event("token", token)

            yield sse_event("done", {"status": "complete"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        def error_stream():
            yield sse_event("error", {"message": str(e)})

        return StreamingResponse(error_stream(), media_type="text/event-stream")
