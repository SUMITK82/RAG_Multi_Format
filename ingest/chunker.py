"""
Turn a DocumentJSON (unified schema) into LangChain Document chunks.

Design choice: we chunk structure-aware rather than blindly splitting raw
text. Table rows / CSV rows stay as single atomic chunks (splitting a row
mid-way destroys its meaning); paragraphs longer than the size threshold
get split with RecursiveCharacterTextSplitter. Every chunk keeps enough
metadata (source_file, page/row, block type) to support citations later.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .schema import DocumentJSON

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_document(doc_json: DocumentJSON) -> List[Document]:
    chunks: List[Document] = []

    for block in doc_json.content:
        base_metadata = {
            "source_file": doc_json.source_file,
            "file_type": doc_json.file_type,
            "block_type": block.type,
            "page": block.page,
            "row": block.row,
        }

        if block.type == "table_row":
            # Keep rows atomic — don't split them further.
            chunks.append(Document(page_content=block.text, metadata=base_metadata))
            continue

        if len(block.text) <= _splitter._chunk_size:
            chunks.append(Document(page_content=block.text, metadata=base_metadata))
        else:
            for piece in _splitter.split_text(block.text):
                chunks.append(Document(page_content=piece, metadata=dict(base_metadata)))

    return chunks


def chunk_documents(doc_jsons: List[DocumentJSON]) -> List[Document]:
    all_chunks: List[Document] = []
    for doc_json in doc_jsons:
        all_chunks.extend(chunk_document(doc_json))
    return all_chunks
