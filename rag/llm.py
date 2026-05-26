import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable")

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "llama-3.1-8b-instant"

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

    try:

        messages = build_messages(
            question,
            context,
            history
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"LLM Error: {str(e)}"


def stream_llm(question, context, history=None, mode="chat"):

    try:

        messages = build_messages(
            question,
            context,
            history
        )

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            stream=True
        )

        for chunk in stream:

            if chunk.choices:

                delta = chunk.choices[0].delta.content

                if delta:
                    yield delta

    except Exception as e:

        yield f"\n\nLLM Streaming Error: {str(e)}"