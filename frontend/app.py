import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://ai-doc-backend-4dvz.onrender.com")

st.set_page_config(page_title="RAG Chat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False


# --------------------
# UPLOAD
# --------------------
file = st.file_uploader("Upload PDF")

if file:
    r = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )

    if r.status_code == 200:
        st.session_state.uploaded = True
        st.success("Uploaded!")
    else:
        st.error(r.text)


# --------------------
# CHAT HISTORY
# --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# --------------------
# ALWAYS VISIBLE INPUT (FIX IMPORTANT)
# --------------------
if st.session_state.uploaded:

    question = st.chat_input("Ask about your document...")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        r = requests.post(
            f"{API_URL}/chat",
            json={"question": question}
        )

        if r.status_code == 200:
            answer = r.json()["answer"]
        else:
            answer = "Error: backend failed"

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)

else:
    st.info("Please upload a PDF first")