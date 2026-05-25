import streamlit as st
import requests
import json

API = "https://ai-doc-backend-4dvz.onrender.com"

# =========================
# INIT STATE
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# UPLOAD
# =========================
def upload(file):

    r = requests.post(
        f"{API}/upload",
        files={"file": file}
    )

    if r.status_code != 200:
        st.error(r.text)
        return

    data = r.json()

    # 🔥 IMPORTANT FIX: set state BEFORE rerun
    st.session_state.profile = data.get("profile")
    st.session_state.filename = data.get("filename")

    st.success("Uploaded successfully")
    st.rerun()


# =========================
# CHAT
# =========================
def chat(question):

    r = requests.post(
        f"{API}/chat",
        json={"question": question}
    )

    if r.status_code != 200:
        return "Error from server"

    return r.json().get("answer", "No answer")


# =========================
# UI
# =========================
st.title("📄 RAG Document Q&A")

# -------------------------
# UPLOAD SECTION
# -------------------------
file = st.file_uploader("Upload PDF")

if file and st.session_state.profile is None:
    upload(file)

# -------------------------
# SHOW DOC STATUS
# -------------------------
if st.session_state.profile:

    st.success(f"Document loaded: {st.session_state.filename}")

    st.write("Word count:", st.session_state.profile.get("word_count"))

    # -------------------------
    # CHAT INPUT
    # -------------------------
    q = st.chat_input("Ask anything about your document")

    if q:

        st.session_state.messages.append(("user", q))

        answer = chat(q)

        st.session_state.messages.append(("ai", answer))

        for role, msg in st.session_state.messages:

            with st.chat_message(role):
                st.write(msg)

else:
    st.warning("Please upload a document first.")