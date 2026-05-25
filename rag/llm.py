import os

def ask_llm(question, context, history=None, mode="Analyze"):

    try:
        # TEMP MOCK (กัน backend crash)
        return f"[MOCK ANSWER]\n\nQuestion: {question}\n\nBased on context: {context[:200]}"

    except Exception as e:
        return f"LLM error: {str(e)}"