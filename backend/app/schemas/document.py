"""
Pydantic schemas for the /api/documents endpoints.

Why a separate schemas layer?
FastAPI uses Pydantic models to validate incoming data and to serialize
outgoing responses. Keeping them in their own file (instead of inline in
the route function) means:
  - The API's data "contract" is documented in one obvious place.
  - Services and routes can share the same shape without copy-pasting it.
  - FastAPI auto-generates accurate OpenAPI docs (visit /docs) from these.
"""

from datetime import datetime
from pydantic import BaseModel


class DocumentOut(BaseModel):
    """What we tell the frontend about one uploaded document."""

    document_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime

    # Filled in from Phase 2 onward; None until the doc has been processed.
    status: str = "uploaded"  # "uploaded" | "processing" | "indexed" | "failed"
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    """What POST /api/documents/upload returns."""

    document: DocumentOut
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str
