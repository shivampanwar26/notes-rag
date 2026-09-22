"""
Document storage service.

This is the ONLY piece of code that knows how documents are physically
stored on disk. The API layer (api/documents.py) never touches the
filesystem directly — it calls functions here. This separation matters
because in a later project you could swap "save to local disk + JSON
index" for "save to S3 + Postgres" by rewriting only this file.

Storage scheme:
  backend/data/documents/
      <document_id>.pdf        <- the actual uploaded file, renamed
      _index.json               <- metadata for every uploaded document

Why rename the file to <document_id>.pdf instead of keeping the
original name on disk?
  - Two students can both upload "notes.pdf" without overwriting each other.
  - The original filename is preserved in metadata for display purposes.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config import settings

logger = logging.getLogger("app.document_store")

_INDEX_PATH = settings.DOCUMENTS_DIR / "_index.json"
_lock = Lock()  # guards read-modify-write of the JSON index


def _read_index() -> dict:
    if not _INDEX_PATH.exists():
        return {}
    with open(_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_index(index: dict) -> None:
    with open(_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, default=str)


def hash_bytes(content: bytes) -> str:
    """SHA-256 of the raw file bytes — used for duplicate-upload detection."""
    return hashlib.sha256(content).hexdigest()


def find_by_hash(file_hash: str) -> dict | None:
    """
    Return an existing document record with the same content hash, if any.
    This is how we detect "you already uploaded this exact PDF" even if
    it's re-uploaded under a different filename.
    """
    with _lock:
        index = _read_index()
    for record in index.values():
        if record.get("file_hash") == file_hash:
            return record
    return None


def save_document(filename: str, content: bytes) -> dict:
    """
    Save an uploaded PDF's bytes to disk and register it in the index.
    Returns the metadata record that was stored. Status starts as
    "uploaded" — the caller (the upload route) is responsible for kicking
    off processing and moving it to "processing" / "indexed" / "failed".
    """
    document_id = str(uuid.uuid4())
    dest_path: Path = settings.DOCUMENTS_DIR / f"{document_id}.pdf"

    with open(dest_path, "wb") as f:
        f.write(content)

    record = {
        "document_id": document_id,
        "filename": filename,
        "size_bytes": len(content),
        "file_hash": hash_bytes(content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "uploaded",
        "page_count": None,
        "chunk_count": None,
        "error": None,
    }

    with _lock:
        index = _read_index()
        index[document_id] = record
        _write_index(index)

    logger.info(f"[UPLOAD] {filename} -> {document_id}.pdf ({len(content)} bytes)")
    return record


def update_status(
    document_id: str,
    status: str,
    page_count: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """
    Move a document through uploaded -> processing -> indexed | failed.
    Called by the background processing pipeline (services/document_processor.py)
    so the frontend can poll GET /api/documents and see progress without
    the upload request itself having to block until indexing finishes.
    """
    with _lock:
        index = _read_index()
        record = index.get(document_id)
        if record is None:
            return
        record["status"] = status
        if page_count is not None:
            record["page_count"] = page_count
        if chunk_count is not None:
            record["chunk_count"] = chunk_count
        record["error"] = error
        index[document_id] = record
        _write_index(index)


def list_documents() -> list[dict]:
    with _lock:
        index = _read_index()
    # Newest first
    return sorted(index.values(), key=lambda d: d["uploaded_at"], reverse=True)


def get_document(document_id: str) -> dict | None:
    with _lock:
        index = _read_index()
    return index.get(document_id)


def delete_document(document_id: str) -> bool:
    """Remove the PDF file and its metadata entry. Returns False if not found."""
    with _lock:
        index = _read_index()
        record = index.pop(document_id, None)
        if record is None:
            return False
        _write_index(index)

    file_path = settings.DOCUMENTS_DIR / f"{document_id}.pdf"
    if file_path.exists():
        file_path.unlink()

    logger.info(f"[DELETE] {record['filename']} ({document_id})")
    return True


def document_path(document_id: str) -> Path:
    return settings.DOCUMENTS_DIR / f"{document_id}.pdf"
