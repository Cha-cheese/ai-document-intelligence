import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are an advanced AI Document Intelligence Assistant.

Your responsibilities:
- Answer naturally like ChatGPT
- Analyze uploaded documents deeply
- Explain clearly and professionally
- Use document context as your primary source
- Maintain conversational flow
- Provide reasoning and intelligent analysis

RULES:
- Never dump raw chunks directly
- Never repeat context blindly
- Summarize and explain intelligently
- If user greets casually, respond naturally
- If information is missing, say:
  "The uploaded document does not contain enough information."

You are a professional AI assistant.
"""


def build_messages(question, context, history=None):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if history:
        for item in history[-6:]:

            messages.append({
                "role": "user",
                "content": item["user"]
            })

            messages.append({
                "role": "assistant",
                "content": item["assistant"]
            })

    prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a detailed, intelligent, and conversational answer.
"""

    messages.append({
        "role": "user",
        "content": prompt
    })

    return messages


def ask_llm(question, context, history=None, mode="chat"):

    messages = build_messages(
        question,
        context,
        history
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=1200
    )

    return response.choices[0].message.content


def stream_llm(question, context, history=None, mode="chat"):

    messages = build_messages(
        question,
        context,
        history
    )

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
        stream=True
    )

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta