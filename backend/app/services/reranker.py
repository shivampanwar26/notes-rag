"""
Reranking (Phase 8) — optional, disabled by default.

CONCEPT — why might vector search return imperfect results?
Embedding models like all-MiniLM-L6-v2 encode a question and a chunk
*independently* into vectors, then compare them with cosine similarity.
That's fast (it's why vector search scales to thousands of chunks), but
it means the model never actually looks at the question and the chunk
side-by-side — it's comparing two separate summaries. Two chunks can end
up with similar vector-search scores even though, read together with the
question, one is obviously the better match and the other is only
superficially related (same topic area, wrong specific answer).

CONCEPT — what reranking does differently.
A cross-encoder reranker takes the (question, chunk) pair *together* as
one input and directly scores "how well does this chunk answer this
question", instead of comparing two independently-computed vectors. This
is much more accurate — but also much slower, because you can't
precompute it ahead of time (it needs the specific question), and you
have to run it once per candidate chunk. That's why it's a *second*
stage: cheap vector search first narrows thousands of chunks down to a
small candidate set (RERANK_CANDIDATES, e.g. 20), then the expensive but
accurate reranker picks the best few from just those 20.

CONCEPT — why keep retrieval and reranking as separate stages instead of
one step?
Different jobs, different cost profiles. Vector search must be cheap
enough to run over the *entire* collection; reranking only needs to run
over a small shortlist. Separating them lets each do what it's good at:
recall (don't miss the right chunk) from vector search, precision (put
the best chunk first) from reranking.

This module is imported lazily by retriever.py and the CrossEncoder model
is loaded on first use — so if ENABLE_RERANKER=false (the default), this
extra model is never downloaded or loaded at all, and the app has one
fewer thing that can go wrong.
"""

import logging

from app.config import settings

logger = logging.getLogger("app.reranker")

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(settings.RERANKER_MODEL)
    return _cross_encoder


def rerank(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Re-score `chunks` for relevance to `question` using a cross-encoder,
    and return the best `top_n`, sorted best-first.
    """
    if not chunks:
        return chunks

    model = _get_cross_encoder()
    pairs = [[question, c["text"]] for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["score"] = round(float(score), 4)  # overwrite with the more accurate score

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_n]
