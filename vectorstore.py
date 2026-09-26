"""
Build / save / load a FAISS vector index over LangChain Documents.

Embeddings run locally via sentence-transformers (all-MiniLM-L6-v2) so the
project has zero embedding API cost and no external dependency for that
step — only the generation step calls the Claude API.
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_index(documents: List[Document], embeddings: Optional[HuggingFaceEmbeddings] = None) -> FAISS:
    embeddings = embeddings or get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def save_index(index: FAISS, path: str = "faiss_index") -> None:
    index.save_local(path)


def load_index(path: str = "faiss_index", embeddings: Optional[HuggingFaceEmbeddings] = None) -> FAISS:
    embeddings = embeddings or get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def index_exists(path: str = "faiss_index") -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "index.faiss"))
