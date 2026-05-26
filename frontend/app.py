import streamlit as st
import requests

API_URL = "https://ai-doc-backend-4dvz.onrender.com"

st.set_page_config(
    page_title="AI Document Chat",
    layout="wide"
)

# =========================
# SESSION
# =========================
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# TITLE
# =========================
st.title("📄 AI Document Chat")

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    files = {
        "file": uploaded_file
    }

    try:

        r = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=180
        )

        r.raise_for_status()

        data = r.json()

        st.success(
            f"Uploaded: {data['filename']}"
        )

        st.session_state.uploaded = True

    except Exception as e:

        st.error(str(e))

# =========================
# CHAT HISTORY
# =========================
for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT
# =========================
prompt = st.chat_input(
    "Ask about the document..."
)

# IMPORTANT:
# ช่องพิมจะอยู่ตลอด
# ไม่ disable แล้ว

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    try:

        r = requests.post(
            f"{API_URL}/chat",
            json={
                "question": prompt
            },
            timeout=180
        )

        r.raise_for_status()

        answer = r.json()["answer"]

    except Exception as e:

        answer = str(e)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })