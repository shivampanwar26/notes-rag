"""
/api/documents routes.

RULE we're following throughout this project: routes stay thin.
A route function's job is only to:
  1. Validate the HTTP-level shape of the request (FastAPI + Pydantic do this).
  2. Call a service function to do the actual work.
  3. Shape the result into a response schema.

All real logic (saving files, validating PDFs, talking to the vector DB,
calling the LLM, ...) lives in app/services/*. This keeps business logic
testable and reusable outside of HTTP (e.g. from a script or a test).
"""

import logging

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.schemas.document import DeleteResponse, DocumentOut, UploadResponse
from app.services import document_processor, document_store, vector_store

logger = logging.getLogger("app.api.documents")

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB — generous for lecture-note PDFs


@router.get("", response_model=list[DocumentOut])
def get_documents():
    """List all uploaded documents, newest first."""
    return document_store.list_documents()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a single PDF upload, validate it, store it, and kick off
    processing (extraction -> chunking -> embedding -> indexing) in the
    background so this request returns immediately.

    Validation performed here (HTTP-level, "is this a plausible upload?"):
      - filename must end in .pdf
      - content-type should be application/pdf
      - file must not be empty / not over the size limit
      - magic-byte check that it's really a PDF
      - duplicate check (same file content already uploaded)

    Deeper validation ("is this actually readable, non-scanned text?")
    happens inside document_processor.process_document, since that's a
    document-processing concern, not an upload concern — it runs after
    this response has already gone back to the browser, so failures there
    surface as status="failed" on the document, not as an HTTP error here.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Some browsers send octet-stream for PDFs; we already checked the
        # extension above, so we only hard-reject content types that are
        # clearly wrong (e.g. image/png, text/plain).
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected content type '{file.content_type}'. Expected a PDF.",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    # Minimal magic-byte sanity check: real PDFs start with "%PDF-"
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400, detail="File does not look like a valid PDF."
        )

    duplicate = document_store.find_by_hash(document_store.hash_bytes(content))
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail=f"This PDF was already uploaded as '{duplicate['filename']}'.",
        )

    record = document_store.save_document(file.filename, content)
    document_store.update_status(record["document_id"], status="processing")
    record["status"] = "processing"

    background_tasks.add_task(document_processor.process_document, record["document_id"])

    return UploadResponse(
        document=DocumentOut(**record),
        message=f"'{file.filename}' uploaded — processing started.",
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    existed = document_store.delete_document(document_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Document not found.")
    # Also remove its vectors from ChromaDB — otherwise search results
    # would keep citing a PDF that no longer exists on disk.
    vector_store.delete_document(document_id)
    return DeleteResponse(document_id=document_id, message="Document deleted.")
