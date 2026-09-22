"""
/api/chat routes (Phase 5/6) and a debug-only retrieval route (Phase 4).

Kept thin, same rule as documents.py: routes call services, they don't
implement the pipeline themselves.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse, RetrievedChunkOut, SourceOut
from app.services import llm, retriever
from app.services.llm import LLMError

logger = logging.getLogger("app.api.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _dedupe_sources(chunks: list[dict]) -> list[SourceOut]:
    """
    Multiple retrieved chunks can come from the same page (e.g. two
    adjacent chunks that both landed on page 42). Citations should list
    each (document, page) once, keeping its best score, sorted by page
    so they read naturally ("Page 42", "Page 43", ...).
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


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Full RAG pipeline: question -> retrieval -> context -> LLM -> answer + sources.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    logger.info(f"[QUERY] {question}")

    chunks = retriever.retrieve(question, document_id=request.document_id)
    logger.info(f"[RETRIEVE] {len(chunks)} chunks")

    if not chunks:
        # Nothing relevant enough was found — tell the user honestly
        # instead of calling the LLM with no real grounding.
        logger.info(f"[DONE] {time.time() - start:.2f}s (no relevant chunks)")
        return ChatResponse(
            answer="I couldn't find the answer in the uploaded notes.",
            sources=[],
        )

    try:
        logger.info("[LLM] Generating response")
        answer = llm.generate_answer(question, chunks)
    except LLMError as e:
        logger.error(f"[LLM] Failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    sources = _dedupe_sources(chunks)
    logger.info(f"[DONE] {time.time() - start:.2f}s")

    return ChatResponse(answer=answer, sources=sources)


@router.post("/debug-retrieve", response_model=list[RetrievedChunkOut])
def debug_retrieve(request: ChatRequest):
    """
    Retrieval only — no LLM call. Lets you inspect exactly which chunks
    a question matches (and their similarity scores) while debugging or
    tuning TOP_K / RETRIEVAL_THRESHOLD, without needing an LLM API key
    configured at all.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retriever.retrieve(question, document_id=request.document_id)
    return [RetrievedChunkOut(**c) for c in chunks]
