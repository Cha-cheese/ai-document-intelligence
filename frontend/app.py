import html
import json
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "https://ai-doc-backend-4dvz.onrender.com"
)

DEFAULT_MODE = "Analyze"


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Ask My Document",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================
# CSS
# =========================================
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

.hero h1 {
    color: var(--ink);
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.hero p {
    color: var(--muted);
}

.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.doc-name {
    font-weight: 700;
    font-size: 1rem;
}

.doc-meta {
    color: var(--muted);
    margin-top: 0.3rem;
}

.ready {
    margin-top: 0.7rem;
    color: var(--green);
    background: var(--green-soft);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    display: inline-block;
    font-weight: 700;
}

.stChatMessage {
    border-radius: 14px;
}

.empty-chat {
    background: white;
    border: 1px solid var(--line);
    padding: 1rem;
    border-radius: 14px;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# =========================================
# SESSION STATE
# =========================================
def init_state():

    defaults = {
        "messages": [],
        "profile": None,
        "filename": None,
        "chunks": 0,
        "pending_question": None,
        "uploaded_once": False
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# =========================================
# HISTORY
# =========================================
def history_payload():

    return [
        {
            "role": m["role"],
            "content": m["content"]
        }
        for m in st.session_state.messages[-10:]
    ]


# =========================================
# STREAM CHAT
# =========================================
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
                current_event = line.replace(
                    "event: ",
                    "",
                    1
                )
                continue

            if not line.startswith("data: "):
                continue

            data = json.loads(
                line.replace("data: ", "", 1)
            )

            if current_event == "token":

                answer += data
                yield answer, False

            elif current_event == "error":

                yield data.get(
                    "message",
                    "AI processing error occurred."
                ), True

            elif current_event == "done":

                yield answer, True


# =========================================
# UPLOAD DOCUMENT
# =========================================
def upload_document(uploaded_file):

    with st.spinner("Reading document..."):

        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file},
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

    if "error" in data:
        raise Exception(data["error"])

    st.session_state.filename = data.get("filename")
    st.session_state.chunks = data.get("chunks", 0)
    st.session_state.profile = data.get("profile")
    st.session_state.messages = []
    st.session_state.uploaded_once = True


# =========================================
# HEADER
# =========================================
def render_header():

    st.markdown("""
    <div class="hero">
        <h1>Ask questions about your document</h1>
        <p>
            Upload a PDF and ask anything about it.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================
# UPLOAD UI
# =========================================
def render_upload():

    st.markdown("""
    <div class="card">
        <h3>Upload PDF</h3>
        <p>Resume, report, contract, notes, research paper.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="main_uploader"
    )

    if uploaded_file is not None:

        if (
            st.session_state.filename != uploaded_file.name
            or st.session_state.profile is None
        ):

            try:

                upload_document(uploaded_file)

                st.success(
                    "Document is ready. You can ask a question now."
                )

                st.rerun()

            except Exception as error:

                st.error(f"Upload failed: {error}")


# =========================================
# DOCUMENT SUMMARY
# =========================================
def render_document_summary():

    profile = st.session_state.profile

    if not profile:
        return

    filename = st.session_state.filename

    word_count = profile.get("word_count", 0)

    reading_time = profile.get(
        "reading_time_minutes",
        1
    )

    document_type = profile.get(
        "document_type",
        "Document"
    )

    st.markdown(f"""
    <div class="card">
        <div class="doc-name">
            {html.escape(filename)}
        </div>

        <div class="doc-meta">
            {html.escape(document_type)}
            · {word_count} words
            · {reading_time} min read
        </div>

        <div class="ready">
            Ready
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================
# CHAT HISTORY
# =========================================
def render_chat_history():

    if not st.session_state.messages:

        st.markdown("""
        <div class="empty-chat">
            Your answers will appear here.
        </div>
        """, unsafe_allow_html=True)

        return

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# =========================================
# ANSWER QUESTION
# =========================================
def answer_question(question):

    if not st.session_state.profile:

        st.error("No document uploaded yet.")

        return

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

                answer_placeholder.markdown(
                    final_answer + "▌"
                )

                if done:
                    break

            if not final_answer.strip():

                final_answer = "No answer generated."

            answer_placeholder.markdown(final_answer)

        except Exception as error:

            final_answer = f"Error: {error}"

            answer_placeholder.error(final_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })


# =========================================
# APP
# =========================================
init_state()

render_header()

# DEBUG
st.write(
    "DEBUG PROFILE:",
    st.session_state.profile
)

if st.session_state.profile:

    render_document_summary()

    if st.button("Clear chat"):

        st.session_state.messages = []

        st.rerun()

else:

    render_upload()

render_chat_history()

question = st.chat_input(
    "Ask anything about the document...",
    disabled=not st.session_state.profile
)

if question:

    answer_question(question)