import html
import json
import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "https://ai-doc-backend-4dvz.onrender.com")
DEFAULT_MODE = "Analyze"


st.set_page_config(
    page_title="Ask My Document",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
:root {
    --bg: #f7f8fb;
    --card: #ffffff;
    --ink: #111827;
    --muted: #64748b;
    --line: #e2e8f0;
    --blue: #2563eb;
    --blue-dark: #1d4ed8;
    --blue-soft: #eff6ff;
    --green: #15803d;
    --green-soft: #ecfdf5;
}

header,
#MainMenu,
footer,
[data-testid="stSidebar"],
[data-testid="stToolbar"],
[data-testid="stHeader"] {
    display: none;
    height: 0;
}

.stApp {
    background:
        linear-gradient(180deg, #eef4ff 0, #f7f8fb 18rem),
        var(--bg);
    color: var(--ink);
}

.block-container {
    max-width: 940px;
    padding: 3rem 1.25rem 7rem;
}

h1, h2, h3, p {
    letter-spacing: 0;
}

.hero {
    margin-bottom: 1.2rem;
}

.hero h1 {
    color: var(--ink);
    font-size: 2.15rem;
    line-height: 1.12;
    font-weight: 820;
    margin: 0;
}

.hero p {
    color: var(--muted);
    font-size: 1.02rem;
    margin: 0.55rem 0 0;
    max-width: 720px;
}

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.15rem;
    box-shadow: 0 16px 44px rgba(15, 23, 42, 0.06);
}

.upload-card {
    margin: 1rem 0;
}

.card-title {
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 760;
    margin-bottom: 0.25rem;
}

.card-copy {
    color: var(--muted);
    font-size: 0.94rem;
    margin-bottom: 0.8rem;
}

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    padding: 0.75rem;
}

[data-testid="stFileUploader"] button {
    background: var(--blue);
    color: #ffffff;
    border: 1px solid var(--blue-dark);
    border-radius: 9px;
    font-weight: 720;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--blue-dark);
    color: #ffffff;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #ffffff;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--muted);
}

.doc-card {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin: 1rem 0;
}

.doc-name {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 760;
    word-break: break-word;
}

.doc-meta {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.22rem;
}

.ready {
    color: var(--green);
    background: var(--green-soft);
    border: 1px solid #bbf7d0;
    border-radius: 999px;
    padding: 0.28rem 0.6rem;
    font-size: 0.82rem;
    font-weight: 720;
    white-space: nowrap;
}

.quick-title {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 720;
    margin: 1rem 0 0.45rem;
}

div.stButton > button {
    background: #ffffff;
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 10px;
    min-height: 2.6rem;
    font-weight: 650;
}

div.stButton > button:hover {
    border-color: #93c5fd;
    color: var(--blue-dark);
}

.stChatMessage {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
}

.stChatMessage,
.stChatMessage *,
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] * {
    color: var(--ink);
}

.stChatMessage p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--ink);
}

.empty-chat {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem;
    color: var(--muted);
    margin-top: 1rem;
}

[data-testid="stChatInput"] {
    background: #f7f8fb;
    border-top: 1px solid var(--line);
}

[data-testid="stChatInput"] textarea {
    color: var(--ink);
    background: #ffffff;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280;
    opacity: 1;
}

[data-testid="stChatInput"] button {
    background: var(--blue);
    color: #ffffff;
}

[data-testid="stChatInput"] button * {
    color: #ffffff;
}

[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] {
    background: #f7f8fb;
}

.small-action {
    margin-top: 0.3rem;
}

@media (max-width: 760px) {
    .hero h1 {
        font-size: 1.72rem;
    }

    .doc-card {
        display: block;
    }

    .ready {
        display: inline-block;
        margin-top: 0.65rem;
    }
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages": [],
        "profile": None,
        "filename": None,
        "chunks": 0,
        "pending_question": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def history_payload():
    return [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages[-10:]
    ]


def stream_chat(question):
    payload = {
        "question": question,
        "mode": DEFAULT_MODE,
        "history": history_payload()
    }
    answer = ""

    with requests.post(
        f"{API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=180
    ) as response:
        response.raise_for_status()
        current_event = None

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line.replace("event: ", "", 1)
                continue

            if not line.startswith("data: "):
                continue

            data = json.loads(line.replace("data: ", "", 1))

            if current_event == "token":
                answer += data
                yield answer, False
            elif current_event == "error":
                yield data.get("message", "AI processing error occurred."), True
            elif current_event == "done":
                yield answer, True


def upload_document(uploaded_file):
    with st.spinner("Reading and indexing your document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file},
            timeout=180
        )
        response.raise_for_status()
        data = response.json()

    st.session_state.filename = data.get("filename")
    st.session_state.chunks = data.get("chunks", 0)
    st.session_state.profile = data.get("profile", {})
    st.session_state.messages = []


def render_header():
    st.markdown("""
    <div class="hero">
        <h1>Ask questions about your document</h1>
        <p>Upload a PDF and get clear, useful answers. The assistant keeps track of the conversation, so follow-up questions work naturally.</p>
    </div>
    """, unsafe_allow_html=True)


def render_upload():
    st.markdown("""
    <div class="card upload-card">
        <div class="card-title">Upload your PDF</div>
        <div class="card-copy">Resume, report, contract, notes, or research paper. PDF only.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file.name != st.session_state.filename:
        try:
            upload_document(uploaded_file)
            st.success("Document is ready. You can ask a question now.")
        except Exception as error:
            st.error(f"Could not connect to the backend: {error}")


def render_document_summary():
    profile = st.session_state.profile

    if not profile:
        return

    filename = st.session_state.filename or "Current document"
    document_type = profile.get("document_type", "Document")
    word_count = profile.get("word_count", 0)
    read_time = profile.get("reading_time_minutes", 1)

    st.markdown(f"""
    <div class="card doc-card">
        <div>
            <div class="doc-name">{html.escape(filename)}</div>
            <div class="doc-meta">{html.escape(document_type)} · {word_count} words · about {read_time} min read</div>
        </div>
        <div class="ready">Ready</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Use another document"):
        uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])

        if uploaded_file and uploaded_file.name != st.session_state.filename:
            try:
                upload_document(uploaded_file)
                st.rerun()
            except Exception as error:
                st.error(f"Could not connect to the backend: {error}")


def render_quick_questions():
    if not st.session_state.profile:
        return

    st.markdown("<div class='quick-title'>Suggested questions</div>", unsafe_allow_html=True)

    prompts = [
        "Summarize this document",
        "What are the key points?",
        "What should I focus on?",
        "What are the risks or gaps?"
    ]
    columns = st.columns(4)

    for column, prompt in zip(columns, prompts):
        with column:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()


def render_chat_history():
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-chat">
            Your answers will appear here. Try asking for a summary, strengths, gaps, or next steps.
        </div>
        """, unsafe_allow_html=True)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def answer_question(question):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        final_answer = ""

        try:
            for partial_answer, done in stream_chat(question):
                final_answer = partial_answer
                answer_placeholder.markdown(f"{final_answer}_")

                if done:
                    break

            answer_placeholder.markdown(final_answer or "No answer generated.")

        except Exception as error:
            final_answer = f"Error: {error}"
            answer_placeholder.error(final_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })


init_state()
render_header()

if st.session_state.profile:
    render_document_summary()

    if st.button("Clear chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()

    render_quick_questions()
else:
    render_upload()

render_chat_history()

question = st.chat_input(
    "Ask anything about the document...",
    disabled=False
)

if st.session_state.pending_question:
    queued_question = st.session_state.pending_question
    st.session_state.pending_question = None
    answer_question(queued_question)
    st.rerun()

if question:
    answer_question(question)
