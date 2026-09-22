"""
Document processing pipeline orchestrator.

This ties together the whole left-hand side of the architecture diagram:

    PDF -> PyMuPDF extraction -> chunking -> embeddings -> ChromaDB

It's called as a FastAPI BackgroundTask right after a successful upload
(see api/documents.py), so the upload request returns immediately and
processing happens without blocking the browser. The document's `status`
field (tracked in document_store's JSON index) is how the frontend knows
when indexing finishes: "processing" -> "indexed" or "failed".

Why a separate orchestrator file instead of putting this logic in the
route? Because this exact sequence also needs to run from the evaluation
script (Phase 9) without going through HTTP at all — keeping it as a
plain function makes it reusable from anywhere.
"""

import logging
import time

from app.services import chunker, document_store, embeddings, pdf_loader, vector_store
from app.services.pdf_loader import PDFProcessingError

logger = logging.getLogger("app.document_processor")


def process_document(document_id: str) -> None:
    record = document_store.get_document(document_id)
    if record is None:
        logger.error(f"[INDEX] Unknown document_id {document_id}")
        return

    filename = record["filename"]
    pdf_path = document_store.document_path(document_id)
    start = time.time()

    try:
        pages = pdf_loader.extract_pages(pdf_path)

        chunks = chunker.chunk_pages(document_id, filename, pages)
        if not chunks:
            raise PDFProcessingError("No text chunks could be produced from this PDF.")

        texts = [c["text"] for c in chunks]
        vectors = embeddings.embed_texts(texts)

        vector_store.add_chunks(chunks, vectors)

        document_store.update_status(
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
        document_store.update_status(document_id, status="failed", error=str(e))
        logger.error(f"[INDEX] Failed {filename}: {e}")

    except Exception as e:
        # Catches embedding failures, ChromaDB failures, or anything
        # unexpected — a document should never get stuck silently in
        # "processing" forever.
        document_store.update_status(
            document_id, status="failed", error=f"Unexpected error: {e}"
        )
        logger.exception(f"[INDEX] Unexpected failure processing {filename}")
