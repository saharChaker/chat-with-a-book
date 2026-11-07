"""Web chat interface for Chat with a Book (Streamlit)."""
import streamlit as st

from src.load_book import load_book
from src.chunker import chunk_text
from src.vector_store import VectorStore
from src.rag import answer_question

st.set_page_config(page_title="Chat with a Book", page_icon="📚")

st.title("📚 Chat with a Book")
st.caption("Ask anything about Frankenstein — answers come from the book's actual text (RAG).")


@st.cache_resource  # load the index ONCE, reuse for every question
def get_store() -> VectorStore:
    chunks = chunk_text(load_book("book/frankenstein.pdf"))
    store = VectorStore()
    store.load(chunks)
    return store


store = get_store()

# Chat history lives in session state so it survives page refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the book..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the book..."):
            answer = answer_question(store, question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})