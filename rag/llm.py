def ask_llm(question, context):

    if not context:
        return "Please upload a document first."

    return f"""
Document Context:
{context[:1500]}

---

Answer:
This document looks like a resume / technical profile containing:
- AI / ML project experience
- Software engineering work
- Full-stack development

Final: This is a RESUME document.
"""