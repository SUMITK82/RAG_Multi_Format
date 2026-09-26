"""
Streamlit UI for the multi-format RAG pipeline.

Flow:
1. Upload one or more files (PDF, DOCX, CSV, XLSX, TXT, MD)
2. Each file is parsed into the unified JSON schema (viewable in the UI —
   this is the project's differentiator, so we surface it, not hide it)
3. JSON content is chunked and embedded into an in-memory FAISS index
4. Ask a question -> retrieve top-k chunks -> LLM generates a cited answer
"""

import os
import tempfile

from dotenv import load_dotenv
import streamlit as st

from ingest import parse_any, chunk_documents
from vectorstore import build_index
from rag_chain import answer_question

load_dotenv()

st.set_page_config(page_title="Multi-Format RAG", layout="wide")
st.title("📄 Multi-Format RAG: any file → unified JSON → cited Q&A")

with st.sidebar:
    st.header("⚙️ Model Setup")

    providers = ["Groq (Free)", "Google Gemini (Free)", "Anthropic Claude (Paid)"]
    default_provider_index = 0 if os.environ.get("GROQ_API_KEY") else 1

    provider = st.selectbox(
        "Select Provider",
        providers,
        index=default_provider_index,
    )

    if provider == "Groq (Free)":
        env_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        help_url = "https://console.groq.com/"
        key_label = "Groq API Key"
        placeholder = "gsk_..."
        model_options = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    elif provider == "Google Gemini (Free)":
        env_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        help_url = "https://aistudio.google.com/"
        key_label = "Google Gemini API Key"
        placeholder = "AIzaSy..."
        model_options = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    else:
        env_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        help_url = "https://console.anthropic.com/"
        key_label = "Anthropic API Key"
        placeholder = "sk-ant-..."
        model_options = ["claude-sonnet-5", "claude-3-5-haiku-20241022"]

    default_key = "" if env_key in ("", "your-key-here", "sk-ant-api03-...") or env_key.endswith("...") else env_key
    api_key = st.text_input(
        key_label,
        type="password",
        value=default_key,
        placeholder=placeholder,
    )
    st.markdown(f"[👉 **Get your free key here**]({help_url})")

    selected_model = st.selectbox("Model", model_options)
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=3)
    st.caption("Supported file types: PDF, DOCX, CSV, XLSX, TXT, MD")

if "index" not in st.session_state:
    st.session_state.index = None
if "doc_jsons" not in st.session_state:
    st.session_state.doc_jsons = []

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "csv", "xlsx", "xls", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Ingest & Index"):
    doc_jsons = []
    with st.spinner("Parsing files into unified JSON..."):
        for uf in uploaded_files:
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            doc_json = parse_any(tmp_path)
            doc_json.source_file = uf.name  # keep the original filename
            doc_jsons.append(doc_json)
            os.unlink(tmp_path)

    with st.spinner("Chunking and building FAISS index (local embeddings)..."):
        chunks = chunk_documents(doc_jsons)
        index = build_index(chunks)

    st.session_state.index = index
    st.session_state.doc_jsons = doc_jsons
    st.success(f"Indexed {len(chunks)} chunks from {len(doc_jsons)} file(s).")

if st.session_state.doc_jsons:
    with st.expander("🔍 View unified JSON output (per file)"):
        for doc_json in st.session_state.doc_jsons:
            st.subheader(doc_json.source_file)
            st.json(doc_json.to_dict())

st.divider()
st.subheader("Ask a question")
question = st.text_input("Your question")

if st.button("Ask") and question:
    key_clean = api_key.strip() if api_key else ""
    is_valid_key = bool(
        key_clean
        and key_clean not in ("", "your-key-here", "sk-ant-api03-...")
        and not key_clean.endswith("...")
    )
    if st.session_state.index is None:
        st.warning("Please upload and index at least one file first.")
    elif not is_valid_key:
        st.warning(f"Please enter a valid {key_label} in the sidebar.")
    else:
        try:
            with st.spinner(f"Retrieving context and generating answer with {selected_model}..."):
                answer, chunks = answer_question(
                    st.session_state.index,
                    question,
                    k=k,
                    provider=provider,
                    model=selected_model,
                    api_key=key_clean,
                )
            st.markdown("### Answer")
            st.write(answer)

            with st.expander("📚 Retrieved sources"):
                for i, c in enumerate(chunks, start=1):
                    meta = c.metadata
                    loc = meta.get("page") or meta.get("row") or "n/a"
                    st.markdown(f"**[{i}] {meta.get('source_file')}** (loc: {loc})")
                    st.code(c.page_content)
        except Exception as exc:
            st.error(f"API Error ({provider}): {exc}")