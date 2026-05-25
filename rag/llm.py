import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def build_messages(question, context, history=None):

    messages = [
        {
            "role": "system",
            "content": f"""
You are an AI document assistant.

Answer ONLY using the provided context.

If the answer is not in the context,
say you cannot find the answer.

CONTEXT:
{context}
"""
        }
    ]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": question
    })

    return messages


def ask_llm(
    question,
    context,
    history=None,
    mode="Analyze"
):

    messages = build_messages(
        question,
        context,
        history
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content


def stream_llm(
    question,
    context,
    history=None,
    mode="Analyze"
):

    messages = build_messages(
        question,
        context,
        history
    )

    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.3,
        stream=True
    )

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta