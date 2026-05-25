import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(question, context, history=None, mode="Analyze"):

    system_prompt = """
You are an AI Document Assistant using RAG.

Rules:
- Use provided context first
- If context is irrelevant, answer generally
- Be natural like ChatGPT
- Do not hallucinate facts not in context
"""

    user_prompt = f"""
Question:
{question}

Context:
{context if context else "No context available"}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


def stream_llm(*args, **kwargs):
    # optional future streaming
    yield "Streaming not enabled yet"