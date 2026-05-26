import streamlit as st
import requests

API_URL = "https://ai-doc-backend-4dvz.onrender.com"

st.set_page_config(page_title="RAG Chat", layout="wide")

# -------------------------
# INIT STATE (IMPORTANT FIX)
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False


# -------------------------
# UPLOAD
# -------------------------
st.title("📄 AI Document Chat")

file = st.file_uploader("Upload PDF")

if file and not st.session_state.uploaded:

    r = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )

    if r.status_code == 200:
        st.success("Uploaded successfully")
        st.session_state.uploaded = True
    else:
        st.error(r.text)


# -------------------------
# CHAT HISTORY (ALWAYS VISIBLE)
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# INPUT (FIX: NEVER DISAPPEAR)
# -------------------------
question = st.chat_input("Ask about your document...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    r = requests.post(
        f"{API_URL}/chat",
        json={"question": question}
    )

    if r.status_code != 200:
        answer = "Error: backend failed"
    else:
        answer = r.json()["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)