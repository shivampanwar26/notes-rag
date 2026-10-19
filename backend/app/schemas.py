from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime
    status: str = "uploaded"
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    document: DocumentOut
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: str | None = None
    include_trace: bool = False


class SourceOut(BaseModel):
    document: str
    page: int
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    trace: dict[str, Any] | None = None


class RetrievedChunkOut(BaseModel):
    text: str
    source: str
    page: int
    chunk_id: str
    score: float | None = None  