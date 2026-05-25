import streamlit as st
import requests

API = "https://ai-doc-backend-4dvz.onrender.com"

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("📄 RAG Chat System")


# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload PDF")

if file:

    r = requests.post(
        f"{API}/upload",
        files={"file": file}
    )

    data = r.json()

    if data.get("ok"):

        st.session_state.session_id = data["session_id"]   # 🔥 IMPORTANT FIX

        st.success("Uploaded successfully!")


# =========================
# CHAT HISTORY
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# =========================
# CHAT INPUT (ALWAYS SHOW)
# =========================
q = st.chat_input("Ask anything about your document")

if q:

    if not st.session_state.session_id:
        st.error("Please upload a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": q})

    r = requests.post(
        f"{API}/chat",
        json={
            "question": q,
            "session_id": st.session_state.session_id
        }
    )

    answer = r.json().get("answer", "error")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()