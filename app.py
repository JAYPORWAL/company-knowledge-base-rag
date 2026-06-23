import streamlit as st
import traceback
from pathlib import Path
from loguru import logger

from config.settings import get_settings
from config.logging_config import configure_logging
from utils.health_check import HealthChecker
from rag.index_builder import IndexBuilder
from rag.query_engine import RAGQueryEngine

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Company Knowledge Base Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. System Bootstrapping (Cached resource)
@st.cache_resource
def bootstrap_system():
    try:
        settings = get_settings()
        configure_logging(settings)
        return settings
    except Exception as e:
        st.error(f"Critical System Initialization Error: {str(e)}")
        st.stop()

settings = bootstrap_system()

# 3. Component Instantiation (Cached to prevent SQLite locks and overhead)
@st.cache_resource
def get_health_checker(settings_obj) -> HealthChecker:
    return HealthChecker(settings_obj)

@st.cache_resource
def get_index_builder(settings_obj) -> IndexBuilder:
    return IndexBuilder(settings_obj)

health_checker = get_health_checker(settings)
index_builder = get_index_builder(settings)

# 4. Premium Responsive Dark Mode CSS Styles
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Overrides */
    .main .block-container {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC !important;
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }

    /* Premium Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #311042 50%, #0F172A 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-title {
        font-size: 2.5rem;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #D8B4FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Badge Indicators */
    .health-badge {
        padding: 5px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid transparent;
        margin-bottom: 1rem;
    }
    .health-healthy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .health-unhealthy {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border-color: rgba(239, 68, 68, 0.3);
    }

    /* Citation block cards */
    .citation-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s, transform 0.2s;
    }
    .citation-card:hover {
        border-color: #C084FC;
        transform: translateY(-1px);
    }
    .citation-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .citation-filename {
        font-weight: 600;
        font-size: 0.95rem;
        color: #F8FAFC;
    }
    .citation-score-badge {
        background-color: rgba(168, 85, 247, 0.15);
        color: #D8B4FE;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .citation-body {
        font-size: 0.9rem;
        color: #CBD5E1;
        line-height: 1.5;
        font-style: italic;
        background-color: #0F172A;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #1E293B;
    }
    .citation-meta-row {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.75rem;
        color: #64748B;
    }

    /* Buttons override */
    .stButton>button {
        background: linear-gradient(90deg, #A855F7 0%, #7E22CE 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton>button:hover {
        opacity: 0.9 !important;
    }

    /* Native chat components customize background */
    [data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 5. Initialize Chat History state variables
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# 6. Sidebar Implementation
with st.sidebar:
    st.markdown("### 🎛️ System Diagnostics")
    diag = health_checker.check_all()
    if diag["status"] == "healthy":
        st.markdown('<span class="health-badge health-healthy">● System Fully Operational</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="health-badge health-unhealthy">● Diagnostics Issue Detected</span>', unsafe_allow_html=True)

    with st.expander("🔍 Health Diagnostics Details"):
        for check_name, check_data in diag["details"].items():
            icon = "🟢" if check_data["healthy"] else "🔴"
            st.markdown(f"{icon} **{check_name.replace('_', ' ').title()}**")
            if not check_data["healthy"]:
                st.caption(f"_{check_data['message']}_")

    st.markdown("---")

    # Ingestion module inside sidebar
    st.markdown("### 📥 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, PPTX, MD):",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "pptx", "md"],
        key="sidebar_uploader"
    )

    if uploaded_files:
        if st.button("Trigger Indexer Pipeline"):
            raw_dir = Path(settings.DATA_RAW_DIR)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            p_bar = st.progress(0)
            status_lbl = st.empty()
            
            success_count = 0
            duplicate_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_lbl.text(f"Processing: {uploaded_file.name}...")
                
                # Sanitize filename to prevent path traversal
                safe_filename = Path(uploaded_file.name).name
                local_file_path = raw_dir / safe_filename
                
                # Write file buffers
                with open(local_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    record = index_builder.ingest_file(local_file_path)
                    if record:
                        success_count += 1
                        st.success(f"Success: {uploaded_file.name} ({record['chunk_count']} nodes)")
                    else:
                        duplicate_count += 1
                        st.warning(f"Skipped duplicate: {uploaded_file.name}")
                except Exception as ex:
                    st.error(f"Error: {uploaded_file.name} - {str(ex)}")
                    logger.error("Ingestion failed: {}", traceback.format_exc())
                
                p_bar.progress((idx + 1) / len(uploaded_files))
            
            status_lbl.text("Ingestion completed.")
            st.balloons()
            st.info(f"Ingested: {success_count} | Skipped: {duplicate_count}")
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")
    
    # Retrieval Tuning Sliders
    top_k = st.slider("Similarity Top K", min_value=1, max_value=10, value=4)
    score_threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    use_hybrid = st.toggle("Enable Hybrid Search (Vector + Text)", value=True)
    
    st.markdown("---")
    st.markdown("### 🗑️ Data Operations")

    col_clear, col_rebuild = st.columns(2)
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []
            st.success("Chat history cleared.")
            st.rerun()
            
    with col_rebuild:
        if st.button("Rebuild Index"):
            with st.spinner("Clearing and rebuilding index..."):
                try:
                    index_builder.rebuild_index()
                    st.success("Index rebuilt successfully.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Rebuild failed: {ex}")

# 7. Main Panel Header
st.markdown("""
<div class="header-banner">
    <div class="header-title">🧠 Company Knowledge Base Portal</div>
    <div class="header-subtitle">Ask questions about internal files. Uses LlamaIndex Hybrid Retrieval with Gemini 2.5 Flash.</div>
</div>
""", unsafe_allow_html=True)

# 8. Check if vector database has documents
index_ready = False
registered_docs = {}
try:
    index = index_builder.get_index()
    registered_docs = index_builder.registry.get_all_registered_documents()
    if registered_docs:
        index_ready = True
except Exception:
    pass

# Tab Layout
tab_chat, tab_catalog = st.tabs(["💬 Interactive Q&A Chat", "📄 Document Catalog"])

with tab_chat:
    if not index_ready:
        st.warning("⚠️ No documents indexed yet. Use the sidebar to upload and ingest documents to start asking questions.")
    else:
        # Render past chat session history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show citations if present
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📚 View Citations", expanded=False):
                        for citation in msg["citations"]:
                            conf = citation["score"] * 100 if use_hybrid else citation["score"] * 100
                            score_label = f"Fused Score: {citation['score']:.4f}" if use_hybrid else f"Match: {conf:.1f}%"
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-head">
                                    <span class="citation-filename">📄 Citation #{citation['citation_number']}: {citation['filename']}</span>
                                    <span class="citation-score-badge">{score_label}</span>
                                </div>
                                <div class="citation-body">
                                    "{citation['snippet']}"
                                </div>
                                <div class="citation-meta-row">
                                    <span><b>Type:</b> {citation['file_type'].upper()}</span>
                                    <span><b>Hash:</b> <code>{citation['sha256_hash'][:16]}</code></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                # Show token counts
                if msg["role"] == "assistant" and msg.get("tokens"):
                    st.caption(
                        f"⚡ Tokens: {msg['tokens']['total_tokens']} "
                        f"(Prompt: {msg['tokens']['prompt_tokens']} | Completion: {msg['tokens']['completion_tokens']})"
                    )

        # Handle user prompt inputs
        if user_prompt := st.chat_input("Ask a question..."):
            # Render user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Save user prompt
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            
            # Render assistant message
            with st.chat_message("assistant"):
                try:
                    # Construct query engine and inject session states to sync memory context
                    engine = RAGQueryEngine(
                        index,
                        similarity_top_k=top_k,
                        score_threshold=score_threshold,
                        hybrid=use_hybrid
                    )
                    
                    # Feed session message history to engine's memory
                    for msg in st.session_state.chat_messages[:-1]:  # skip last prompt we just added
                        engine.memory.add_message(msg["role"], msg["content"])
                    
                    # Execute streaming query
                    wrapper = engine.query_stream(user_prompt)
                    
                    # Render streamed response
                    full_answer = st.write_stream(wrapper.response_generator())
                    
                    # Display Token usage
                    tokens = wrapper.token_usage
                    st.caption(
                        f"⚡ Tokens: {tokens['total_tokens']} "
                        f"(Prompt: {tokens['prompt_tokens']} | Completion: {tokens['completion_tokens']})"
                    )
                    
                    # Display citations
                    if wrapper.citations:
                        with st.expander("📚 View Citations", expanded=False):
                            for citation in wrapper.citations:
                                conf = citation["score"] * 100 if use_hybrid else citation["score"] * 100
                                score_label = f"Fused Score: {citation['score']:.4f}" if use_hybrid else f"Match: {conf:.1f}%"
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-head">
                                        <span class="citation-filename">📄 Citation #{citation['citation_number']}: {citation['filename']}</span>
                                        <span class="citation-score-badge">{score_label}</span>
                                    </div>
                                    <div class="citation-body">
                                        "{citation['snippet']}"
                                    </div>
                                    <div class="citation-meta-row">
                                        <span><b>Type:</b> {citation['file_type'].upper()}</span>
                                        <span><b>Hash:</b> <code>{citation['sha256_hash'][:16]}</code></span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Log message and metadata inside session history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "citations": wrapper.citations,
                        "tokens": tokens
                    })
                    
                except Exception as ex:
                    logger.error("Failed to parse RAG chat request: {}", str(ex))
                    st.error(f"Failed to query knowledge base: {str(ex)}")
                    st.exception(ex)

with tab_catalog:
    st.markdown("### Document Catalog")
    if not registered_docs:
        st.info("No documents are currently indexed in ChromaDB.")
    else:
        st.write("List of documents in the vector database index:")
        
        # Build clean markdown catalog table
        table_content = [
            "| Filename | File Type | Chunks | Upload Timestamp (UTC) | SHA256 Checksum |",
            "| :--- | :--- | :---: | :--- | :--- |"
        ]
        
        for doc in registered_docs.values():
            time_str = doc["upload_timestamp"]
            try:
                time_str = time_str.split(".")[0].replace("T", " ")
            except Exception:
                pass
                
            table_content.append(
                f"| **{doc['filename']}** | {doc['file_type'].upper()} | {doc['chunk_count']} | "
                f"{time_str} | `{doc['sha256_hash'][:16]}...` |"
            )
            
        st.markdown("\n".join(table_content))
