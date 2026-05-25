import streamlit as st
import requests
import json
import os

API_URL = os.getenv(
    "API_URL",
    "https://ai-doc-backend-4dvz.onrender.com"
)

st.set_page_config(page_title="Ask Doc", layout="wide")

# =========================
# STATE
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# UPLOAD SAFE (NO CRASH)
# =========================
def upload(file):

    r = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )

    try:
        data = r.json()
    except:
        st.error("Backend error")
        return

    if r.status_code != 200 or "error" in data:
        st.error(data.get("error", "Upload failed"))
        return

    st.session_state.profile = data["profile"]
    st.session_state.filename = data["filename"]
    st.session_state.messages = []

    st.success("Document uploaded successfully")

# =========================
# CHAT
# =========================
def chat(question):

    payload = {
        "question": question
    }

    r = requests.post(f"{API_URL}/chat", json=payload)

    try:
        data = r.json()
    except:
        st.error("Backend not responding")
        return

    if "error" in data:
        st.error(data["error"])
        return

    return data["answer"]

# =========================
# UI
# =========================
st.title("📄 Ask your document")

file = st.file_uploader("Upload PDF")

if file:
    upload(file)

if st.session_state.profile:
    st.write("✅ Ready:", st.session_state.filename)

    q = st.chat_input("Ask something about document")

    if q:
        st.session_state.messages.append(("user", q))

        answer = chat(q)

        st.session_state.messages.append(("assistant", answer))

        for role, msg in st.session_state.messages:
            with st.chat_message(role):
                st.write(msg)