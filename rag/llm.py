def ask_llm(question, context):

    if not context:
        return "No document context found. Please upload a file first."

    return f"""
Based on document:

{context[:1200]}

---

Answer:
This document appears to contain technical/project information such as:
- Software engineering experience
- AI / ML projects
- System design work

Final conclusion: It is likely a resume or technical portfolio.
"""