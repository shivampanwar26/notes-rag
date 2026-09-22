"""
All HTTP routes: documents (upload / list / delete) and chat (ask / debug).

Routes stay thin — validate the request, call ingest or rag, shape the
response. The pipelines themselves live in ingest.py and rag.py so they can
also run from scripts and tests without going through HTTP.
"""

import logging
import time

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app import ingest, rag
from app.rag import LLMError
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentOut,
    RetrievedChunkOut,
    SourceOut,
    UploadResponse,
)

logger = logging.getLogger("app.api")

documents_router = APIRouter(prefix="/api/documents", tags=["documents"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # generous for lecture-note PDFs


# ---- Documents -------------------------------------------------------


@documents_router.get("", response_model=list[DocumentOut])
def get_documents():
    """List uploaded documents, newest first."""
    return ingest.list_documents()


@documents_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """
    Accept one PDF, validate it, store it, and index it in the background
    so this request returns immediately.

    Checks here are upload-level ("is this a plausible PDF?"). Whether the
    text is actually extractable is decided during processing and surfaces
    as status="failed" on the document, not as an error on this response.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Some browsers send octet-stream for PDFs; the extension was already
    # checked, so only clearly-wrong types are rejected.
    if file.content_type not in ("application/pdf", "application/octet-stream"):
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

    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File does not look like a valid PDF.")

    duplicate = ingest.find_by_hash(ingest.hash_bytes(content))
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail=f"This PDF was already uploaded as '{duplicate['filename']}'.",
        )

    record = ingest.save_document(file.filename, content)
    ingest.update_status(record["document_id"], status="processing")
    record["status"] = "processing"

    background_tasks.add_task(ingest.process_document, record["document_id"])

    return UploadResponse(
        document=DocumentOut(**record),
        message=f"'{file.filename}' uploaded — processing started.",
    )


@documents_router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    if not ingest.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, message="Document deleted.")


# ---- Chat ------------------------------------------------------------


def _dedupe_sources(chunks: list[dict]) -> list[SourceOut]:
    """
    Several retrieved chunks can share a page, so citations are collapsed
    to one entry per (document, page) keeping the best score, ordered so
    they read naturally.
    """
    best: dict[tuple[str, int], float] = {}
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in best or c["score"] > best[key]:
            best[key] = c["score"]

    sources = [
        SourceOut(document=doc, page=page, score=score)
        for (doc, page), score in best.items()
    ]
    sources.sort(key=lambda s: (s.document, s.page))
    return sources


@chat_router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Full RAG pipeline: question -> retrieval -> LLM -> answer + sources."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    logger.info(f"[QUERY] {question}")

    chunks = rag.retrieve(question, document_id=request.document_id)
    logger.info(f"[RETRIEVE] {len(chunks)} chunks")

    if not chunks:
        # Nothing relevant was found — say so rather than asking the LLM to
        # answer from nothing.
        logger.info(f"[DONE] {time.time() - start:.2f}s (no relevant chunks)")
        return ChatResponse(
            answer="I couldn't find the answer in the uploaded notes.", sources=[]
        )

    try:
        logger.info("[LLM] Generating response")
        answer = rag.generate_answer(question, chunks)
    except LLMError as e:
        logger.error(f"[LLM] Failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    logger.info(f"[DONE] {time.time() - start:.2f}s")
    return ChatResponse(answer=answer, sources=_dedupe_sources(chunks))


@chat_router.post("/debug-retrieve", response_model=list[RetrievedChunkOut])
def debug_retrieve(request: ChatRequest):
    """
    Retrieval only, no generation — inspect exactly which chunks a question
    matches and their scores while tuning TOP_K, weights, or thresholds.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = rag.retrieve(question, document_id=request.document_id)
    return [RetrievedChunkOut(**c) for c in chunks]
