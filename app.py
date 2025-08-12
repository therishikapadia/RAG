import streamlit as st
import requests

API_URL = "http://localhost:8000/api"

st.title("RAG Chatbot")

# File upload
st.header("Upload a file")
uploaded_file = st.file_uploader("Choose a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    resp = requests.post(f"{API_URL}/upload_file", files=files)
    if resp.ok:
        data = resp.json()
        st.success(f"Uploaded! Chunks: {data['num_chunks']}")
    else:
        st.error(f"Upload failed: {resp.text}")

st.divider()

# Chatbot UI
st.header("Chat with your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

top_k = st.slider("Number of sources", 1, 10, 5)

def send_message():
    query = st.session_state.chat_input
    if query:
        resp = requests.post(
            f"{API_URL}/query",
            json={"query": query, "top_k": top_k}
        )
        if resp.ok:
            data = resp.json()
            st.session_state.chat_history.append({
                "question": query,
                "answer": data["answer"],
                "sources": data.get("sources", [])
            })
            st.session_state.chat_input = ""  # This is safe here
        else:
            st.error(f"Query failed: {resp.text}")

st.text_input(
    "Type your question and press Enter",
    key="chat_input",
    on_change=send_message
)

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    if chat["sources"]:
        st.markdown(f"*Sources:* {', '.join(chat['sources'])}")
    st.markdown("---")