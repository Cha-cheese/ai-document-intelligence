import streamlit as st
import requests

API = "https://ai-doc-backend-4dvz.onrender.com"

st.set_page_config(page_title="RAG Chat", layout="wide")

# =========================
# SESSION
# =========================
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload PDF")

if file:
    if st.button("Upload"):
        r = requests.post(
            f"{API}/upload",
            files={"file": file}
        )
        r.raise_for_status()

        st.session_state.uploaded = True
        st.success("Uploaded!")

# =========================
# ALWAYS SHOW CHAT INPUT (FIX YOUR BUG)
# =========================
if not st.session_state.uploaded:
    st.warning("Upload document first")

# CHAT UI MUST ALWAYS EXIST
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Ask anything...")  # ✅ FIX: NEVER DISAPPEAR

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    r = requests.post(
        f"{API}/chat",
        json={"question": question}
    )

    if r.status_code != 200:
        st.error(r.text)
    else:
        answer = r.json()["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)