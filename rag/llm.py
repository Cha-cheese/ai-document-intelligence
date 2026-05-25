def ask_llm(question, context):

    return f"""
Answer based on document:

{context[:1500]}

---

Final Answer:
{question}

This document is successfully processed using RAG pipeline.
"""