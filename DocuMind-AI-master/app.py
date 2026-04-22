import os
import shutil
import streamlit as st
from dotenv import load_dotenv

# Local imports
from loader import load_pdfs
from chunking import chunk_documents
from vectorstore import create_vector_store, delete_vector_store
from retriever import get_retriever, format_docs_for_citation
from rag_pipeline import generate_answer

# Load .env file
load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMindAI",
    page_icon="🤖",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .citation-box {
        background: #1e1e2e;
        border-left: 4px solid #7b61ff;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #e2e2e2 !important;      /* explicit light text — prevents black-on-dark */
        line-height: 1.5;
    }
    .citation-box b {
        color: #a78bfa !important;      /* purple accent for source heading */
        display: block;
        margin-bottom: 4px;
    }
    .citation-box i {
        color: #c9c9c9 !important;
        font-style: italic;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .badge-hf  { background: #ff9800; color: #000 !important; }
    .badge-gem { background: #4285f4; color: #fff !important; }
    .status-active {
        color: #4caf50;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-empty {
        color: #9e9e9e;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🤖 DocuMindAI — Strict Document QA Agent")
st.markdown(
    "Upload research PDFs and ask questions. "
    "I answer **strictly** from the document content — no hallucinations."
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
    api_key_input = st.text_input("Google Gemini API Key", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    # ── Gemini Generation Model ───────────────────────────────────────────────
    st.subheader("🧠 Generation Model")
    gemini_model = st.selectbox(
        "Choose Gemini model:",
        options=["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite-preview", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"],
        index=0
    )

    st.markdown("---")

    # ── Embedding Method ──────────────────────────────────────────────────────
    st.subheader("🔢 Embedding Method")
    embedding_method = st.radio(
        "Choose embedding model:",
        options=["huggingface", "gemini"],
        index=1,
        format_func=lambda x: (
            "🟠 HuggingFace — BAAI/bge-small-en (local, free)"
            if x == "huggingface"
            else "🔵 Google Gemini — gemini-embedding-001 (cloud)"
        )
    )

    st.markdown("---")

    # ── Document Upload ───────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF Papers", type=["pdf"], accept_multiple_files=True
    )

    # ── Strategy Toggles ─────────────────────────────────────────────────────
    st.subheader("📐 Chunking Strategy")
    chunk_strategy = st.radio(
        "Select:",
        options=["A", "B"],
        format_func=lambda x: "A — size 500, overlap 100" if x == "A" else "B — size 800, overlap 200",
    )

    st.subheader("🔍 Retrieval Strategy")
    retrieval_strategy = st.radio(
        "Select:",
        options=["similarity", "mmr"],
        format_func=lambda x: (
            "Cosine Similarity" if x == "similarity" else "Max Marginal Relevance (MMR)"
        ),
    )

    # ── Processing Mode ─────────────────────────────────────────────────────
    st.subheader("🔄 Index Mode")
    process_mode = st.radio(
        "When processing new PDFs:",
        options=["replace", "append"],
        format_func=lambda x: (
            "🗑️ Replace — wipe existing chunks and rebuild fresh"
            if x == "replace"
            else "➕ Append — add new chunks to existing index"
        )
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    # ── Processing Logic ─────────────────────────────────────────────────────
    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        elif embedding_method == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            st.error("Gemini embeddings require a Google API Key. Please enter it above.")
        else:
            with st.spinner("Processing documents…"):
                temp_dir = "temp_pdfs"
                os.makedirs(temp_dir, exist_ok=True)
                file_paths = []
                for file in uploaded_files:
                    path = os.path.join(temp_dir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(path)

                # Step 1 — Load
                docs = load_pdfs(file_paths)
                st.write(f"📖 Loaded **{len(docs)}** pages.")

                # Step 2 — Chunk
                chunks = chunk_documents(docs, strategy=chunk_strategy)
                st.write(f"✂️  Created **{len(chunks)}** text chunks.")

                # Step 3 — Embed & Store
                create_vector_store(
                    chunks,
                    embedding_method=embedding_method,
                    mode=process_mode,
                )

                # Show which embedding was used
                badge_class = "badge-hf" if embedding_method == "huggingface" else "badge-gem"
                badge_label = (
                    "HuggingFace BAAI/bge-small-en"
                    if embedding_method == "huggingface"
                    else "Gemini gemini-embedding-001"
                )
                st.markdown(
                    f'<span class="badge {badge_class}">Embedded with: {badge_label}</span>',
                    unsafe_allow_html=True,
                )
                st.success("✅ Vector store created and saved!")

                # Cleanup temp files
                shutil.rmtree(temp_dir)
                # Reset any pending delete confirmation
                st.session_state.pop("confirm_delete", None)

    st.markdown("---")

    # ── Index Status & Delete ─────────────────────────────────────────────────
    st.subheader("🗂️ Index Management")

    index_exists = os.path.exists("faiss_index")

    if index_exists:
        # Read which embedding was used when the index was built
        meta_file = os.path.join("faiss_index", "embedding_method.txt")
        stored_method = "unknown"
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                stored_method = f.read().strip()
        badge_class = "badge-hf" if stored_method == "huggingface" else "badge-gem"
        badge_label = (
            "HuggingFace BAAI/bge-small-en" if stored_method == "huggingface" else "Gemini gemini-embedding-001"
        )
        st.markdown('<span class="status-active">● Index active</span>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="badge {badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-empty">○ No index found</span>', unsafe_allow_html=True)

    # Two-step delete: first click shows confirmation, second click deletes
    if not st.session_state.get("confirm_delete", False):
        if st.button(
            "🗑️ Delete All Stored Data",
            use_container_width=True,
            disabled=not index_exists
        ):
            st.session_state["confirm_delete"] = True
            st.rerun()
    else:
        st.warning("⚠️ This will permanently delete ALL embedded chunks and index data. This cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", use_container_width=True, type="primary"):
                deleted = delete_vector_store()  # uses the dedicated delete function
                st.session_state.pop("confirm_delete", None)
                if deleted:
                    st.success("🗑️ All data deleted — index, chunks, and embeddings removed.")
                else:
                    st.info("No data found to delete.")
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pop("confirm_delete", None)
                st.rerun()

    st.markdown("---")

    # ── Clear Chat History ────────────────────────────────────────────────────
    st.subheader("💬 Chat")
    if st.button("🧹 Clear Chat History", use_container_width=True,
                 disabled=not bool(st.session_state.get("messages"))):
        st.session_state["messages"] = []
        st.rerun()

# ─── Chat UI ──────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("📚 Show Top 3 Sources"):
                for i, cit in enumerate(message["citations"], 1):
                    st.markdown(
                        f'<div class="citation-box">'
                        f"<b>[{i}] {cit['source']}</b> — Page {cit['page']}<br/>"
                        f"<i>{cit['snippet'][:350]}…</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# User input
if prompt := st.chat_input("Ask a question about your uploaded papers…"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please set your Google API Key in the sidebar.")
    elif not os.path.exists("faiss_index"):
        st.error("No index found. Please upload PDFs and click ⚡ Process Documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # Retrieve using saved embedding method (auto-detected from metadata)
                    retriever = get_retriever(strategy=retrieval_strategy, k=3)
                    docs = retriever.invoke(prompt)

                    # Format context + citations
                    formatted_context, citations = format_docs_for_citation(docs)

                    # Generate answer — pass selected model from sidebar
                    chat_history_str = str(st.session_state.messages[-4:-1])
                    answer = generate_answer(
                        context=formatted_context,
                        question=prompt,
                        chat_history=chat_history_str,
                        model=gemini_model,
                    )

                    st.markdown(answer)

                    # Citations block
                    if citations and "I don't know" not in answer:
                        with st.expander("📚 Top 3 Supporting Sources", expanded=True):
                            for i, cit in enumerate(citations, 1):
                                st.markdown(
                                    f'<div class="citation-box">'
                                    f"<b>[{i}] {cit['source']}</b> — Page {cit['page']}<br/>"
                                    f"<i>{cit['snippet'][:350]}…</i>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations if "I don't know" not in answer else None,
                    })

                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        st.error(
                            "⚠️ **Gemini Rate Limit Reached (429)**\n\n"
                            "Your free-tier quota for **`" + gemini_model + "`** is exhausted.\n\n"
                            "**Fix:** Switch to **`gemini-2.0-flash-lite`** or **`gemini-1.5-flash`** "
                            "in the **🧠 Generation Model** selector in the sidebar, then try again."
                        )
                    else:
                        st.error(f"Error: {e}")

