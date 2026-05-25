import streamlit as st
import requests

API = "https://ai-doc-backend-4dvz.onrender.com"

st.title("📄 RAG Document Q&A")

# upload
file = st.file_uploader("Upload PDF")

if file:

    r = requests.post(
        f"{API}/upload",
        files={"file": file}
    )

    if r.status_code == 200:
        st.success("Uploaded")
        st.session_state["uploaded"] = True
        st.session_state["file"] = file.name
    else:
        st.error(r.text)

# chat state fix
if "uploaded" not in st.session_state:
    st.warning("Upload file first")
    st.stop()

q = st.chat_input("Ask anything")

if q:

    res = requests.post(
        f"{API}/chat",
        json={"question": q}
    )

    if res.status_code == 200:
        st.write(res.json()["answer"])
    else:
        st.error(res.text)