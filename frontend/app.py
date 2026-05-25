import html
import json
import os
import requests
import streamlit as st

# =========================
# API CONFIG (FIXED)
# =========================
API_URL = os.getenv("API_URL", "https://ai-doc-backend-4dvz.onrender.com").rstrip("/")

DEFAULT_MODE = "Analyze"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Ask My Document",
    page_icon="AI",
    layout="wide"
)

# =========================
# INIT STATE
# =========================
def init_state():
    defaults = {
        "messages": [],
        "profile": None,
        "filename": None,
        "uploaded": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================
# HISTORY
# =========================
def history_payload():
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-8:]
    ]

# =========================
# STREAM CHAT (ROBUST)
# =========================
def stream_chat(question):

    payload = {
        "question": question,
        "history": history_payload(),
        "mode": DEFAULT_MODE
    }

    try:
        with requests.post(
            f"{API_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=180
        ) as r:

            r.raise_for_status()

            event = None
            answer = ""

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                try:
                    data = json.loads(line.replace("data: ", ""))
                except:
                    continue

                if event == "token":
                    answer += data
                    yield answer, False

                elif event == "done":
                    yield answer, True
                    return

                elif event == "error":
                    yield data.get("message", "Server error"), True
                    return

    except Exception as e:
        yield f"Connection error: {str(e)}", True

# =========================
# UPLOAD
# =========================
def upload_document(file):

    with st.spinner("Uploading..."):

        r = requests.post(
            f"{API_URL}/upload",
            files={"file": file},
            timeout=180
        )

        r.raise_for_status()
        data = r.json()

    if "error" in data:
        raise Exception(data["error"])

    st.session_state.profile = data.get("profile", {})
    st.session_state.filename = data.get("filename")
    st.session_state.messages = []
    st.session_state.uploaded = True

# =========================
# UI
# =========================
def render_header():
    st.title("📄 Ask My Document")
    st.caption("Upload PDF → Chat with it")

def render_upload():

    st.subheader("Upload PDF")

    file = st.file_uploader("PDF only", type=["pdf"])

    if file:

        if st.session_state.filename != file.name:

            try:
                upload_document(file)
                st.success("Upload success")
                st.rerun()

            except Exception as e:
                st.error(f"Upload failed: {e}")

def render_doc():

    p = st.session_state.profile
    if not p:
        return

    st.info(
        f"{st.session_state.filename} | "
        f"{p.get('word_count',0)} words"
    )

def chat_ui():

    if not st.session_state.profile:
        st.warning("Upload document first")
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask something...")

    if q:

        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):

            box = st.empty()
            final = ""

            for text, done in stream_chat(q):
                final = text
                box.markdown(text + "▌")
                if done:
                    break

            box.markdown(final)

        st.session_state.messages.append({"role": "assistant", "content": final})

# =========================
# APP
# =========================
init_state()
render_header()
render_upload()
render_doc()
chat_ui()