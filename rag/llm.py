import os

from groq import Groq


def _client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it in your Render environment variables."
        )

    return Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are an advanced AI document analyst.

Your job:
- deeply analyze documents
- infer meaning from the retrieved context when the evidence supports it
- explain reasoning clearly but concisely
- answer professionally
- avoid generic refusal answers

If the document is a resume:
- analyze strengths
- analyze weaknesses
- evaluate career suitability
- score professionally when useful
- explain why, using evidence from the resume

If the user asks for an opinion:
- provide thoughtful analysis
- support it with evidence from the document
- distinguish direct evidence from reasonable inference

Always:
- use the retrieved context
- cite important evidence by source number when source numbers are provided
- prioritize the most important facts, insights, and evidence
- avoid filler, repetition, and overly long explanations
- sound intelligent, practical, and professional

Default answer style:
- be concise unless the user explicitly asks for detail
- answer in 1 short paragraph or 3-6 focused bullets
- include only the reasoning needed to justify the answer
- do not over-explain obvious points
- do not write long sections such as Strengths, Weaknesses, Score, and Reasoning
  unless they are directly useful for the question

Only say that the document does not provide enough information when the answer is
truly impossible from the retrieved context and no reasonable inference can be made.
"""


def _wants_detailed_answer(question):
    detailed_keywords = [
        "detail",
        "detailed",
        "deep",
        "in-depth",
        "explain",
        "analysis",
        "analyze",
        "reasoning",
        "why",
        "ละเอียด",
        "อธิบาย",
        "วิเคราะห์",
        "เจาะลึก",
        "เหตุผล",
        "ทำไม"
    ]
    normalized_question = question.lower()
    return any(keyword in normalized_question for keyword in detailed_keywords)


MODE_INSTRUCTIONS = {
    "Analyze": "Analyze the document like a sharp business or research analyst.",
    "Summarize": "Summarize only the most important points. Be brief and practical.",
    "Recruiter Review": "Review the document like a professional recruiter. Focus on fit, strengths, gaps, and role suitability.",
    "Research Assistant": "Act as a research assistant. Extract claims, evidence, methods, findings, and implications.",
    "Legal Analysis": "Analyze carefully and conservatively. Flag obligations, risks, missing details, and ambiguous wording. Do not provide legal advice.",
    "Explain Like Beginner": "Explain in simple language with minimal jargon.",
    "Technical Review": "Review technical quality, architecture, tools, implementation depth, and risks."
}


def _format_history(history):
    if not history:
        return "No previous conversation."

    recent_messages = history[-8:]
    formatted_messages = []

    for message in recent_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if not content:
            continue

        formatted_messages.append(f"{role.upper()}: {content}")

    return "\n".join(formatted_messages) or "No previous conversation."


def _build_prompt(question, context, history=None, mode="Analyze"):
    answer_style = (
        "The user asked for explanation or analysis, so answer with clear reasoning "
        "but still keep it focused and avoid unnecessary length."
        if _wants_detailed_answer(question)
        else
        "Answer briefly and directly. Focus only on the important points. "
        "Use 1 short paragraph or 3-5 concise bullets. Do not make the answer long."
    )
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["Analyze"])
    conversation_history = _format_history(history)

    return f"""
MODE:
{mode}

MODE INSTRUCTION:
{mode_instruction}

RECENT CONVERSATION:
{conversation_history}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER STYLE:
{answer_style}

Use the document evidence first, then add careful inference if appropriate.
When source numbers are present, cite key claims with the relevant source number.
"""


def ask_llm(question, context, history=None, mode="Analyze"):
    prompt = _build_prompt(question, context, history, mode)

    response = _client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def stream_llm(question, context, history=None, mode="Analyze"):
    prompt = _build_prompt(question, context, history, mode)

    response = _client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content

        if content:
            yield content
