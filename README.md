# Multi-Format RAG

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=SUMITK82/RAG_Multi_Format&branch=main&mainModule=app.py)

Turn documents in different formats into one searchable knowledge base, then ask questions and get answers grounded in the source material.

This project demonstrates the complete retrieval-augmented generation (RAG) flow: file parsing, a shared document representation, structure-aware chunking, local vector search, and cited answers from a selectable language model.

## What it does

- **Ingests** PDF, DOCX, CSV, XLSX, TXT, and Markdown files.
- **Normalizes** parsed content into a shared `DocumentJSON` schema, with source metadata and page or row references where available.
- **Chunks** long text while keeping table rows together so tabular records retain their context.
- **Indexes locally** with FAISS and `sentence-transformers/all-MiniLM-L6-v2`.
- **Retrieves relevant passages** and asks a selected LLM to answer from those passages with source citations.
- **Evaluates retrieval** with a small hit-rate@k script and a user-provided set of expected source files.

## How the pipeline works

```text
PDF / DOCX / CSV / XLSX / TXT / MD
                 |
                 v
     Format-specific parsers
                 |
                 v
       Unified DocumentJSON
                 |
                 v
 Structure-aware chunks + metadata
                 |
                 v
 Local embeddings -> FAISS search
                 |
       question -> top-k passages
                 |
                 v
 Gemini / Groq / Claude answer + citations
```

Embeddings are generated locally. Answer generation uses the selected provider's API, so an API key and internet access are needed to ask questions.

## Project layout

```text
rag-multi-format/
├── app.py                         # Streamlit interface and upload-to-answer workflow
├── ingest/
│   ├── __init__.py                # Public ingestion helpers
│   ├── parsers.py                 # File-format parsers and extension dispatch
│   ├── schema.py                  # DocumentJSON and ContentBlock models
│   └── chunker.py                 # Structure-aware LangChain document chunking
├── vectorstore.py                 # Local embeddings and FAISS index operations
├── rag_chain.py                   # Retrieval, provider selection, and cited answers
├── eval.py                        # Source-based retrieval hit-rate@k evaluation
├── eval_questions.example.json    # Example evaluation question format
├── requirements.txt               # Python dependencies
└── .env.example                   # Example API-key environment variable
```

## Requirements

- Python 3.10 or newer
- An API key for one supported answer-generation provider: **Groq**, **Google Gemini**, or **Anthropic**

The first indexing run downloads the local embedding model, which may take a little time and disk space.

## Deploy a public demo

Click **Deploy to Streamlit** at the top of this README and sign in to Streamlit Community Cloud. Select this repository, the `main` branch, and `app.py` as the main file, then deploy. Once it finishes, Streamlit provides a public `*.streamlit.app` URL. Add that URL to the repository's **About → Website** field to give visitors a direct link to the running app. The deploy button starts the deployment flow; it is not itself a live app URL.

The app accepts the selected provider's API key in its sidebar, so you do not need to commit credentials or add them to GitHub. The public app uses its hosting account's resources; avoid uploading confidential documents.

## Setup on Windows

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the key for your chosen provider to `.env`:

```dotenv
GROQ_API_KEY=your-groq-key
```

Or use `GEMINI_API_KEY=your-gemini-key` or `ANTHROPIC_API_KEY=your-anthropic-key`. You can also enter the key in the app's sidebar. Keep `.env` private and do not commit real API keys.

If PowerShell blocks virtual-environment activation, either allow scripts for the current user according to your organization's policy or run the app with the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Start the app locally

With the virtual environment activated:

```powershell
streamlit run app.py
```

Once Streamlit is running, open **[http://localhost:8501](http://localhost:8501)** on the same computer. This local link is clickable from the repository README, but it only works while the app is running on your computer; GitHub cannot start a process on your machine.

Select a provider and model, upload files, and click **Ingest & Index**. You can inspect the normalized JSON for each uploaded file, then ask questions. The app shows the retrieved passages alongside the answer.

The uploaded files and FAISS index are held in the current Streamlit session; the app does not persist the index between sessions.

## Evaluate retrieval

Evaluation checks whether the expected source file appears among the top `k` retrieved chunks. It measures **source hit-rate@k**, not answer correctness.

1. Copy `eval_questions.example.json` to `eval_questions.json` and replace the sample questions with questions about your own documents. Set `expected_source` to the filename used when indexing.
2. Build and save an index from those same documents. For example, place supported files in a `documents` folder and run this from the project root:

   ```powershell
   @'
   from pathlib import Path
   from ingest import parse_any, chunk_documents
   from vectorstore import build_index, save_index

   documents = [
       parse_any(str(path))
       for path in Path("documents").iterdir()
       if path.is_file() and path.suffix.lower() in {
           ".pdf", ".docx", ".csv", ".xlsx", ".txt", ".md"
       }
   ]
   chunks = chunk_documents(documents)
   save_index(build_index(chunks), "faiss_index")
   print(f"Saved an index with {len(chunks)} chunks from {len(documents)} files.")
   '@ | .\.venv\Scripts\python.exe -
   ```

3. Run the evaluation:

   ```powershell
   .\.venv\Scripts\python.exe eval.py --index faiss_index --questions eval_questions.json --k 3
   ```

`eval.py` loads a previously saved index. The index-building step above is separate from the Streamlit workflow, which keeps its index in memory.

## Current scope and limitations

- PDF extraction reads embedded text; scanned PDFs need OCR, which is not included.
- Legacy `.xls` files are accepted by the upload interface, but reading them requires the optional `xlrd` package, which is not currently listed in `requirements.txt`.
- Retrieval uses dense FAISS similarity search, without keyword/hybrid search or reranking.
- Citation locations are available when the parser can associate content with a PDF page or tabular row; other formats may only provide the source filename.
- The evaluation script measures whether retrieval found the expected file, not whether the generated answer is factually correct.
