import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are an AI document assistant.

Answer the user's question using ONLY the provided context.

If the answer is not found in the context, say:
"I could not find that information in the document."

Be clear and concise.
"""


def build_messages(question, context, history=None):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if history:

        for msg in history[-6:]:

            if (
                isinstance(msg, dict)
                and "role" in msg
                and "content" in msg
            ):

                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

    messages.append({
        "role": "user",
        "content":
        f"""
DOCUMENT CONTEXT:

{context}

QUESTION:
{question}
"""
    })

    return messages


def ask_llm(question, context, history=None, mode="Analyze"):

    try:

        messages = build_messages(
            question,
            context,
            history
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )

        answer = response.choices[0].message.content

        if not answer:
            return "No answer generated."

        return answer

    except Exception as e:

        print("LLM ERROR:", str(e))

        return f"LLM Error: {str(e)}"


def stream_llm(question, context, history=None, mode="Analyze"):

    try:

        messages = build_messages(
            question,
            context,
            history
        )

        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=700,
            stream=True
        )

        for chunk in stream:

            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    except Exception as e:

        print("STREAM ERROR:", str(e))

        yield f"\n\nError: {str(e)}"