"""
The ingestion pipeline: everything between "a PDF is uploaded" and "its
chunks are searchable", in one file.

    PDF -> page-by-page text -> chunks -> embeddings -> ChromaDB

Text is extracted per page and the page number rides along through every
later step, because citations are the point: "explained on page 42" has to
be traceable to a real page, and a whole-document blob loses that forever.

`process_document` runs as a FastAPI background task so uploads return
immediately, and the document's status (uploaded -> processing -> indexed |
failed) is how the UI follows along.
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import rag
from app.config import settings

logger = logging.getLogger("app.ingest")


class PDFProcessingError(Exception):
    """Raised for any PDF that can't be turned into usable text."""


# ══════════════════════════════════════════════════════════════════════
# 1. PDF TEXT EXTRACTION
#
# PyMuPDF is fast, needs no system dependencies, and returns clean text
# for the text-based PDFs lecture notes and slides almost always are.
# ══════════════════════════════════════════════════════════════════════


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Return [(page_number, text)] with pages 1-indexed, the way people refer
    to them.

    Fails loudly on a corrupt PDF, an empty PDF, or a pure scan with no text
    layer — answers can't be grounded in text that was never read.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFProcessingError(f"Could not open PDF (file may be corrupted): {e}")

    if doc.page_count == 0:
        doc.close()
        raise PDFProcessingError("PDF has no pages.")

    pages = [(i + 1, doc.load_page(i).get_text("text")) for i in range(doc.page_count)]
    doc.close()

    if sum(len(text.strip()) for _, text in pages) == 0:
        raise PDFProcessingError(
            "No extractable text found. This PDF may be a scanned image "
            "with no text layer — OCR is not supported yet."
        )

    logger.info(f"[PARSE] {len(pages)} pages")
    return pages


# ══════════════════════════════════════════════════════════════════════
# 2. CHUNKING
#
# Embedding a whole PDF into one vector would average dozens of unrelated
# topics into something close to no question. Chunks are small enough to
# each carry one idea, with overlap so a concept explained across a chunk
# boundary still appears whole somewhere.
#
# Splitting happens per page so every chunk keeps its page number, and
# RecursiveCharacterTextSplitter breaks on paragraphs before sentences
# before words, rather than slicing mid-word every N characters.
# ══════════════════════════════════════════════════════════════════════


def chunk_pages(
    document_id: str, filename: str, pages: list[tuple[int, str]]
) -> list[dict]:
    """Turn (page, text) pairs into chunk dicts carrying citation metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for page_number, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue  # cover pages and section dividers produce nothing

        for i, piece in enumerate(splitter.split_text(page_text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": f"{document_id}-p{page_number}-c{i}",
                    "text": piece,
                    "document_id": document_id,
                    "source": filename,
                    "page": page_number,
                }
            )

    logger.info(f"[CHUNK] {len(chunks)} chunks")
    return chunks


# ══════════════════════════════════════════════════════════════════════
# 3. DOCUMENT STORE
#
# The only code that knows how documents sit on disk:
#
#     backend/data/documents/<document_id>.pdf   the uploaded file
#     backend/data/documents/_index.json         metadata for all of them
#
# Files are renamed to their id so two "notes.pdf" uploads can coexist;
# the original name is kept in metadata for display and citations.
# ══════════════════════════════════════════════════════════════════════

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
    return hashlib.sha256(content).hexdigest()


def find_by_hash(file_hash: str) -> dict | None:
    """Detect a re-upload of identical content, even under a new filename."""
    with _lock:
        index = _read_index()
    return next(
        (r for r in index.values() if r.get("file_hash") == file_hash), None
    )


def save_document(filename: str, content: bytes) -> dict:
    """Write the PDF to disk, register it, and return its metadata record."""
    document_id = str(uuid.uuid4())
    with open(document_path(document_id), "wb") as f:
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
    """Move a document through uploaded -> processing -> indexed | failed."""
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
        _write_index(index)


def list_documents() -> list[dict]:
    with _lock:
        index = _read_index()
    return sorted(index.values(), key=lambda d: d["uploaded_at"], reverse=True)


def get_document(document_id: str) -> dict | None:
    with _lock:
        index = _read_index()
    return index.get(document_id)


def delete_document(document_id: str) -> bool:
    """Remove the PDF, its metadata, and its vectors. False if unknown id."""
    with _lock:
        index = _read_index()
        record = index.pop(document_id, None)
        if record is None:
            return False
        _write_index(index)

    file_path = document_path(document_id)
    if file_path.exists():
        file_path.unlink()

    # Without this, search would keep citing a PDF that no longer exists.
    rag.delete_document(document_id)

    logger.info(f"[DELETE] {record['filename']} ({document_id})")
    return True


def document_path(document_id: str) -> Path:
    return settings.DOCUMENTS_DIR / f"{document_id}.pdf"


# ══════════════════════════════════════════════════════════════════════
# 4. PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════


def process_document(document_id: str) -> None:
    """Extract, chunk, embed, and index one uploaded PDF."""
    record = get_document(document_id)
    if record is None:
        logger.error(f"[INDEX] Unknown document_id {document_id}")
        return

    filename = record["filename"]
    start = time.time()

    try:
        pages = extract_pages(document_path(document_id))

        chunks = chunk_pages(document_id, filename, pages)
        if not chunks:
            raise PDFProcessingError("No text chunks could be produced from this PDF.")

        rag.add_chunks(chunks, rag.embed_texts([c["text"] for c in chunks]))

        update_status(
            document_id,
            status="indexed",
            page_count=len(pages),
            chunk_count=len(chunks),
        )
        logger.info(
            f"[INDEX] Completed {filename} "
            f"({len(pages)} pages, {len(chunks)} chunks, {time.time() - start:.2f}s)"
        )

    except PDFProcessingError as e:
        update_status(document_id, status="failed", error=str(e))
        logger.error(f"[INDEX] Failed {filename}: {e}")

    except Exception as e:
        # Embedding or ChromaDB failures must not leave a document stuck
        # in "processing" forever.
        update_status(document_id, status="failed", error=f"Unexpected error: {e}")
        logger.exception(f"[INDEX] Unexpected failure processing {filename}")
