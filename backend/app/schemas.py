"""
Pydantic models for every endpoint — the API's data contract in one place.

FastAPI uses these to validate requests, serialize responses, and generate
the OpenAPI docs at /docs.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ---- Documents -------------------------------------------------------


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime
    status: str = "uploaded"  # uploaded | processing | indexed | failed
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    document: DocumentOut
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ---- Chat ------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Optionally scope retrieval to one uploaded document.
    document_id: str | None = None


class SourceOut(BaseModel):
    """One citation: which PDF and page a piece of the answer came from."""

    document: str
    page: int
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class RetrievedChunkOut(BaseModel):
    """Used by the debug-retrieve endpoint — retrieval with no LLM call."""

    text: str
    source: str
    page: int
    score: float
    chunk_id: str
