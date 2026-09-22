"""
Pydantic schemas for the /api/chat endpoints (Phase 5/6).
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Optional: scope retrieval to a single uploaded document instead of
    # searching across everything the user has uploaded.
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
    """Used by the debug-retrieve endpoint (Phase 4) — retrieval with no LLM."""

    text: str
    source: str
    page: int
    score: float
    chunk_id: str
