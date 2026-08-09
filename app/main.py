import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

from app.ingestion.pdf_loader import extract_text_from_pdf

import streamlit as st
from dotenv import load_dotenv

from app.ingestion.pdf_loader import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.llm.groq_client import ask_groq, rewrite_question
from app.agent.decision_agent import classify_question, casual_reply, out_of_scope_reply, is_pdf_answer_sufficient
from app.agent.web_search import search_web
from app.memory.conversation_memory import ConversationMemory
from app.ingestion.pdf_loader import extract_text_from_pdf, get_file_hash

load_dotenv()
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# ---------- Session state (loads once per browser session) ----------
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(collection_name="pdf_chatbot", dimension=384)
if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "bm25_store" not in st.session_state:
    st.session_state.bm25_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.title("📄 PDF Chatbot")

st.markdown("""
<style>
.user-bubble {
    background-color: #3a3a3a;
    color: white;
    padding: 10px 16px;
    border-radius: 16px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 12px;
    text-align: left;
}
.assistant-bubble {
    color: white;
    padding: 6px 0;
    margin-bottom: 16px;
    max-width: 85%;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar: Upload ----------
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files and st.session_state.bm25_store is not None:
                continue

            file_bytes = uploaded_file.getbuffer()
            file_hash = get_file_hash(bytes(file_bytes))

            with st.spinner(f"Processing {uploaded_file.name}..."):
                temp_path = f"data/raw_pdfs/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                pages = extract_text_from_pdf(temp_path, source_name=uploaded_file.name, file_hash=file_hash)
                new_chunks = chunk_pages(pages)

                # Always add to local session state (needed for BM25 + display)
                st.session_state.chunks.extend(new_chunks)

                # Only embed + upload to Qdrant if this exact content isn't already there
                already_in_qdrant = st.session_state.vector_store.hash_exists(file_hash)
                if not already_in_qdrant:
                    embeddings = st.session_state.embedder.embed_chunks(new_chunks)
                    st.session_state.vector_store.add(embeddings, new_chunks)

                st.session_state.bm25_store = BM25Store(st.session_state.chunks)
                st.session_state.processed_files.add(uploaded_file.name)

            st.success(f"✅ {uploaded_file.name} ready — {len(new_chunks)} chunks")

    if st.session_state.processed_files:
        st.markdown("**Uploaded documents:**")
        for fname in st.session_state.processed_files:
            st.markdown(f"- {fname}")

    if st.button("🗑️ Clear conversation"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

# ---------- Chat history display ----------
for turn in st.session_state.chat_history:
    st.markdown(f'<div class="user-bubble">{turn["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-bubble">{turn["answer"]}</div>', unsafe_allow_html=True)
    if turn.get("sources"):
        with st.expander("📚 Sources"):
            for s in turn["sources"]:
                st.markdown(f"- **{s['source_file']}**, page {s['page_number']}")

# ---------- Chat input ----------
question = st.chat_input("Ask a question about your PDF(s)...")

if question:
    if not st.session_state.processed_files or st.session_state.bm25_store is None:
        st.warning("Please upload a PDF first.")
    else:
        st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            category = classify_question(question)
            sources = []

            if category == "CASUAL":
                answer = casual_reply(question)
            elif category == "OUT_OF_SCOPE":
                answer = out_of_scope_reply(question)
            else:
                history_text = st.session_state.memory.get_history_text()
                standalone_q = rewrite_question(question, history_text)

                query_embedding = st.session_state.embedder.embed_query(standalone_q)
                faiss_results = st.session_state.vector_store.search(query_embedding, top_k=10)
                bm25_results = st.session_state.bm25_store.search(standalone_q, top_k=10)
                fused = reciprocal_rank_fusion(faiss_results, bm25_results, top_n=10)
                top_chunks = st.session_state.reranker.rerank(standalone_q, fused, top_n=5)

                pdf_answer = ask_groq(standalone_q, top_chunks)
                answer = pdf_answer

                if not is_pdf_answer_sufficient(standalone_q, pdf_answer):
                    web_info = search_web(standalone_q)
                    if web_info:
                        answer = f"{pdf_answer}\n\n---\n**Additional info (from web, not in document):**\n{web_info}"

                sources = top_chunks

            final_answer = answer if answer and answer.strip() else "Sorry, I couldn't generate a response. Please try rephrasing your question."
            st.markdown(f'<div class="assistant-bubble">{final_answer}</div>', unsafe_allow_html=True)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['source_file']}**, page {s['page_number']}")

            st.session_state.memory.add_turn(question, final_answer)
            st.session_state.chat_history.append({
                "question": question, "answer": final_answer, "sources": sources
            })