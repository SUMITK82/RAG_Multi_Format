"""
Unified document schema.

Every file type (PDF, DOCX, CSV, TXT/MD, ...) gets parsed into this SAME
structure. This is the core design decision of the project: downstream
code (chunking, embedding, retrieval) never needs to know what the
original file type was.
"""

from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    """A single unit of extracted content (a paragraph, a table row, etc.)."""

    type: Literal["paragraph", "heading", "table_row", "raw_text"]
    text: str
    page: Optional[int] = None        # for PDFs
    row: Optional[int] = None         # for CSV/XLSX
    level: Optional[int] = None       # heading level, if applicable
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentJSON(BaseModel):
    """The unified representation produced for every ingested file."""

    source_file: str
    file_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: List[ContentBlock] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()
