"""
Semantic retrieval — the "R" in RAG (Phase 4).

Pipeline:
    question -> question embedding -> ChromaDB similarity search -> top-K chunks

CONCEPT — semantic search vs. keyword search.
A keyword/full-text search for "demand paging" only finds chunks that
literally contain those words. Semantic search compares *meaning*: a
chunk that says "pages are brought into physical memory only when the
process references them" will match well even without the phrase
"demand paging" appearing verbatim, because the embeddings are close in
vector space. This is what lets students ask questions in their own
words instead of guessing the textbook's exact phrasing.

CONCEPT — Top-K retrieval, and why K matters both ways.
K is how many chunks we hand to the LLM as context.
  - Too few (K=1): if the real answer spans two adjacent chunks (common
    near a chunk boundary), the LLM only sees half of it and may answer
    incompletely or say it can't find the answer.
  - Too many (K=20): most of those chunks are irrelevant "noise" sitting
    in the LLM's context window alongside the few good ones. This dilutes
    the prompt, can distract the model into blending unrelated content
    into the answer, costs more tokens/latency, and makes citations messy.
  TOP_K=5 is a reasonable default for short-to-medium questions;
  configurable via .env.

CONCEPT — similarity score and the relevance threshold.
Vector search *always* returns its K nearest neighbors, even if none of
them are actually relevant (e.g. asking a Machine Learning question when
only Operating Systems notes are uploaded — it'll still return the
"closest" OS chunks, just with low similarity). RETRIEVAL_THRESHOLD
filters those out before they ever reach the LLM: chunks scoring below
it are dropped. This is what lets the app honestly say "I couldn't find
the answer in the uploaded notes" instead of forcing an answer out of an
irrelevant chunk.
"""

import logging

from app.config import settings
from app.services import embeddings, reranker, vector_store

logger = logging.getLogger("app.retriever")


def retrieve(
    question: str, top_k: int | None = None, document_id: str | None = None
) -> list[dict]:
    """
    Returns the top-K most relevant chunks for `question`, each as:
        {chunk_id, text, source, page, score}
    sorted by relevance (best first). Chunks below RETRIEVAL_THRESHOLD are
    excluded. Optionally scoped to a single document_id.
    """
    top_k = top_k or settings.TOP_K
    query_vector = embeddings.embed_query(question)

    # When reranking is enabled, cast a wider net first (more candidates),
    # then let the reranker narrow it down to the best top_k. See reranker.py.
    fetch_k = settings.RERANK_CANDIDATES if settings.ENABLE_RERANKER else top_k

    candidates = vector_store.search(query_vector, top_k=fetch_k, document_id=document_id)
    candidates = [c for c in candidates if c["score"] >= settings.RETRIEVAL_THRESHOLD]

    if settings.ENABLE_RERANKER:
        results = reranker.rerank(question, candidates, top_n=top_k)
    else:
        results = candidates[:top_k]

    return results
