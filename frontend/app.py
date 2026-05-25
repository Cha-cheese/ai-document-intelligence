import streamlit as st
import requests

API = "https://ai-doc-backend-4dvz.onrender.com"

# =========================
# STATE INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = None

if "filename" not in st.session_state:
    st.session_state.filename = None


# =========================
# UI HEADER
# =========================
st.title("📄 AI Document Chat (RAG)")

st.caption("Upload document and chat like ChatGPT")


# =========================
# SIDEBAR UPLOAD (เหมือน ChatGPT attach file)
# =========================
with st.sidebar:
    st.header("📎 Upload Document")

    file = st.file_uploader("PDF only", type=["pdf"])

    if file:

        with st.spinner("Uploading..."):

            r = requests.post(
                f"{API}/upload",
                files={"file": file}
            )

        if r.status_code == 200:
            data = r.json()

            st.session_state.profile = data.get("profile")
            st.session_state.filename = data.get("filename")

            st.success("Document loaded")
        else:
            st.error(r.text)


    if st.session_state.profile:
        st.divider()
        st.write("📄", st.session_state.filename)
        st.write("Words:", st.session_state.profile.get("word_count"))


# =========================
# CHAT DISPLAY
# =========================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================
# CHAT INPUT (ALWAYS VISIBLE - FIX สำคัญ)
# =========================
question = st.chat_input("Ask anything about your document...")

if question:

    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # call backend
    try:
        r = requests.post(
            f"{API}/chat",
            json={"question": question}
        )

        if r.status_code == 200:
            answer = r.json().get("answer")

        else:
            answer = "Server error"

    except Exception as e:
        answer = str(e)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)