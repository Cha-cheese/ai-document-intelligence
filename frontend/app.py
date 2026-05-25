import html
import json
import os
import requests
import streamlit as st

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

    st.session_state.profile = data.get("profile", {})
    st.session_state.filename = data.get("filename")
    st.session_state.messages = []

# =========================
# CHAT STREAM
# =========================
def chat_stream(q):
    r = requests.post(
        f"{API_URL}/chat/stream",
        json={"question": q, "history": st.session_state.messages},
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
# UI HEADER
# =========================
st.title("📄 Ask your document")

# =========================
# UPLOAD ALWAYS VISIBLE
# =========================
file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    if file.name != st.session_state.filename:
        upload(file)
        st.success("Uploaded")

# =========================
# SHOW DOC INFO
# =========================
if st.session_state.profile:
    st.info(f"{st.session_state.filename}")

# =========================
# CHAT HISTORY
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT (IMPORTANT FIX)
# =========================
question = st.chat_input("Ask about your document...")

if question:

    if not st.session_state.profile:
        st.error("Upload document first")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box = st.empty()
        final = ""

        try:
            for t in chat_stream(question):
                final = t
                box.markdown(t + "▌")

            box.markdown(final)

        except Exception as e:
            final = f"Error: {e}"
            box.error(final)

    st.session_state.messages.append({"role": "assistant", "content": final})