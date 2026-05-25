import streamlit as st
import requests

API_URL = "https://YOUR-BACKEND.onrender.com"

st.set_page_config(page_title="RAG Chat", layout="wide")

# =========================
# STATE
# =========================
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload PDF")

if file:
    r = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )

    r.raise_for_status()
    data = r.json()

    st.session_state.uploaded = True
    st.success("Uploaded successfully!")

# =========================
# CHAT
# =========================
if not st.session_state.uploaded:
    st.warning("Please upload a document first")
    st.stop()

q = st.chat_input("Ask anything")

if q:
    r = requests.post(
        f"{API_URL}/chat",
        json={"question": q}
    )

    r.raise_for_status()
    ans = r.json()["answer"]

    st.chat_message("user").write(q)
    st.chat_message("assistant").write(ans)