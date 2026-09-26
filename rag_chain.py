"""
Retrieval-augmented generation over the FAISS index, supporting multiple LLM
providers (Google Gemini, Groq, Anthropic Claude). Answers are required to
cite the source file (and page/row) for every claim, and to say "not found in
the documents" rather than guessing when the retrieved context doesn't answer
the question.
"""

import os
from typing import List, Tuple, Optional

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY using the provided context chunks. Every factual claim must be
followed by a citation in square brackets referencing the source, like
[source_file, page 3] or [source_file, row 12].
If the context does not contain the answer, say so plainly instead of
guessing. Do not use outside knowledge."""


def format_context(chunks: List[Document]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        loc = f"page {meta.get('page')}" if meta.get('page') is not None else (
            f"row {meta.get('row')}" if meta.get('row') is not None else "n/a"
        )
        lines.append(
            f"[{i}] source: {meta.get('source_file')} ({loc})\n{chunk.page_content}"
        )
    return "\n\n".join(lines)


def retrieve(index, query: str, k: int = 3) -> List[Document]:
    return index.similarity_search(query, k=k)


def get_llm(provider: str = "Google Gemini (Free)", model: Optional[str] = None, api_key: Optional[str] = None):
    provider_lower = provider.lower()
    if "gemini" in provider_lower or "google" in provider_lower:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        selected_model = model or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=selected_model,
            api_key=key,
            temperature=0,
            max_output_tokens=1000,
        )
    elif "groq" in provider_lower:
        from langchain_groq import ChatGroq
        key = api_key or os.environ.get("GROQ_API_KEY")
        selected_model = model or "openai/gpt-oss-120b"
        return ChatGroq(
            model=selected_model,
            api_key=key,
            temperature=0,
            max_tokens=1000,
        )
    elif "anthropic" in provider_lower or "claude" in provider_lower:
        from langchain_anthropic import ChatAnthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        selected_model = model or "claude-sonnet-5"
        return ChatAnthropic(
            model=selected_model,
            api_key=key,
            temperature=0,
            max_tokens=1000,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def answer_question(
    index,
    query: str,
    k: int = 3,
    provider: str = "Google Gemini (Free)",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    chunks = retrieve(index, query, k=k)
    context = format_context(chunks)

    llm = get_llm(provider=provider, model=model, api_key=api_key)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = llm.invoke(messages)
    return response.content, chunks
