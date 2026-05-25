import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "https://ai-doc-backend-4dvz.onrender.com").rstrip("/")

st.set_page_config(page_title="Doc Chat", layout="wide")

# =========================
# STATE
# =========================
def init():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False

init()

# =========================
# UPLOAD
# =========================
def upload(file):

    r = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )
    r.raise_for_status()

    data = r.json()

    st.session_state.filename = data["filename"]
    st.session_state.profile = data["profile"]
    st.session_state.uploaded = True
    st.session_state.messages = []

# =========================
# STREAM CHAT
# =========================
def stream_chat(q):

    r = requests.post(
        f"{API_URL}/chat/stream",
        json={"question": q},
        stream=True
    )

    r.raise_for_status()

    event = None
    answer = ""

    for line in r.iter_lines(decode_unicode=True):

        if not line:
            continue

        if line.startswith("event:"):
            event = line.split(":")[1].strip()
            continue

        if not line.startswith("data:"):
            continue

        data = json.loads(line.replace("data: ", ""))

        if event == "token":
            answer += data
            yield answer

        if event == "done":
            break

# =========================
# UI
# =========================
st.title("📄 Document Chat")

# =========================
# UPLOAD ALWAYS VISIBLE
# =========================
file = st.file_uploader("Upload PDF", type=["pdf"])

if file and file.name != st.session_state.filename:
    upload(file)
    st.success("Uploaded successfully")
    st.rerun()

# =========================
# DOC INFO
# =========================
if st.session_state.profile:
    st.info(st.session_state.filename)

# =========================
# CHAT HISTORY
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT (FIXED)
# =========================
q = st.chat_input("Ask about document...")

if q:

    if not st.session_state.uploaded:
        st.error("Upload document first")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": q})

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        box = st.empty()
        final = ""

        for t in stream_chat(q):
            final = t
            box.markdown(t + "▌")

        box.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})