"""
AI Research Assistant — Main App
Phase 5 - Streamlit Frontend

Run with: streamlit run app.py
"""

import streamlit as st
import uuid

# ── Page config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from chains.rag_chain import build_chat_chain, build_rag_chain, build_summarize_chain
from vectorstore.vectordb import build_vectorstore, load_vectorstore
from memory.memory_manager import clear_session
from tools.tools import summarize_text, extract_keywords, word_count
from utils.helpers import (
    save_uploaded_pdf,
    format_mode_label,
    truncate,
    format_stats_for_display,
)

# ── Session State (persists across reruns) ────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "mode" not in st.session_state:
    st.session_state.mode = "beginner"

# ── Cached Chains (built once, reused) ────────────────────────────────────────
@st.cache_resource
def get_chat_chain():
    return build_chat_chain()

@st.cache_resource
def get_summarize_chain():
    return build_summarize_chain()

def get_rag_chain(vs):
    return build_rag_chain(vs)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Powered by Groq + LangChain")
    st.divider()

    # ── Mode Toggle ───────────────────────────────────────────────────────────
    st.subheader("🎓 Explanation Mode")
    mode = st.radio(
        "Choose your level",
        ["beginner", "expert"],
        format_func=format_mode_label,
        index=0 if st.session_state.mode == "beginner" else 1,
        help="Beginner = simple language. Expert = technical depth."
    )
    st.session_state.mode = mode
    st.divider()

    # ── PDF Uploader ──────────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    st.caption("Ask questions directly from your PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Your PDFs are processed locally. Nothing is sent to the cloud.",
    )

    if uploaded_files:
        if st.button("🔄 Index Documents", use_container_width=True):
            with st.spinner("Reading and embedding PDFs..."):
                paths = [save_uploaded_pdf(f) for f in uploaded_files]
                st.session_state.vectorstore = build_vectorstore(paths)
            st.success(f"✅ Indexed {len(uploaded_files)} PDF(s)!")

    if st.session_state.vectorstore:
        st.info("📚 Documents indexed — RAG mode active")
    else:
        st.warning("No documents indexed yet")

    st.divider()

    # ── Memory Controls ───────────────────────────────────────────────────────
    st.subheader("🧠 Conversation Memory")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    if st.button("🗑️ Clear Chat & Memory", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · Groq · FAISS · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — 3 Tabs
# ─────────────────────────────────────────────────────────────────────────────
st.title("AI Research Assistant 🔬")

tab_chat, tab_summarize, tab_analyze = st.tabs([
    "💬 Chat",
    "📝 Summarize",
    "🔍 Analyze Text",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:

    # Show mode badge
    if st.session_state.vectorstore:
        st.success("📚 RAG Mode — answering from your uploaded documents")
    else:
        st.info("💬 Chat Mode — general knowledge (upload PDFs to enable RAG)")

    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything..."):

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {
                    "configurable": {
                        "session_id": st.session_state.session_id
                    }
                }

                if st.session_state.vectorstore:
                    # RAG mode — answer from documents
                    chain = get_rag_chain(st.session_state.vectorstore)
                    response = chain.invoke(
                        {
                            "input": prompt,
                            "mode": st.session_state.mode,
                        },
                        config=config,
                    )
                else:
                    # Plain chat mode — general knowledge
                    chain = get_chat_chain()
                    response = chain.invoke(
                        {
                            "input": prompt,
                            "mode": st.session_state.mode,
                            "context_block": "",
                        },
                        config=config,
                    )

            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SUMMARIZE
# ─────────────────────────────────────────────────────────────────────────────
with tab_summarize:
    st.subheader("📝 Summarize Any Text")
    st.caption("Paste any article, paper excerpt, or notes and get a clean summary")

    text_input = st.text_area(
        "Paste your text here",
        height=250,
        placeholder="Paste any research paper, article, or notes here...",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        # Quick tool summary (prose)
        if st.button(
            "✨ Quick Summary",
            use_container_width=True,
            disabled=not text_input.strip(),
        ):
            with st.spinner("Summarizing..."):
                result = summarize_text(text_input, mode=st.session_state.mode)
            st.markdown("### Summary")
            st.markdown(result)

    with col2:
        # Full structured summary via chain
        if st.button(
            "📋 Structured Summary",
            use_container_width=True,
            disabled=not text_input.strip(),
        ):
            with st.spinner("Building structured summary..."):
                chain = get_summarize_chain()
                result = chain.invoke({
                    "text": text_input,
                    "mode": st.session_state.mode,
                })
            st.markdown("### Structured Summary")
            st.markdown(result)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYZE TEXT
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyze:
    st.subheader("🔍 Analyze Any Text")
    st.caption("Get reading stats and extract keywords from any text")

    text_to_analyze = st.text_area(
        "Paste text to analyze",
        height=200,
        placeholder="Paste any text here...",
    )

    if st.button(
        "🔍 Analyze",
        use_container_width=False,
        disabled=not text_to_analyze.strip(),
    ):
        # ── Reading Stats ─────────────────────────────────────────────────────
        stats = word_count(text_to_analyze)

        st.markdown("### 📊 Reading Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Words", stats["words"])
        col2.metric("Sentences", stats["sentences"])
        col3.metric("Paragraphs", stats["paragraphs"])
        col4.metric("Characters", stats["characters"])

        col5, col6 = st.columns(2)
        col5.metric("📖 Read Time", stats["estimated_read_time"])
        col6.metric("🎓 Study Time", stats["estimated_study_time"])

        st.divider()

        # ── Keywords ──────────────────────────────────────────────────────────
        st.markdown("### 🏷️ Top Keywords")
        keywords = extract_keywords(text_to_analyze)

        if keywords:
            # Display as nice pills
            keyword_display = "  ".join(
                f"`{kw}`" for kw in keywords
            )
            st.markdown(keyword_display)
        else:
            st.info("No significant keywords found.")