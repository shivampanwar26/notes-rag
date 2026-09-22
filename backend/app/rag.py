"""
The RAG engine: everything between "a question comes in" and "an answer
goes out", in one file.

    question
      -> multi-query expansion        (llama3 rewrites it 3 ways)
      -> hybrid search per query      (dense vectors + BM25 keywords)
      -> reciprocal rank fusion       (one ranking out of many)
      -> cross-encoder reranking      (precision pass over the shortlist)
      -> grounded generation          (llama3 answers from those chunks only)

WHY THESE FOUR STAGES

Multi-query: a student's phrasing ("how does paging work again") often
shares no vocabulary with the textbook's ("pages are brought into physical
memory on reference"). One query is one guess at the right phrasing;
several queries are several guesses, and fusion keeps whatever they agree on.

Hybrid: dense vectors match meaning but under-weight literal tokens, so
"RAID 5" or "O(log n)" can rank below a vaguely-related paragraph. BM25 is
the opposite — exact terms, no semantics. Running both covers each one's
blind spot.

RRF: the two legs produce incomparable numbers (cosine similarity in [0,1]
vs unbounded BM25 scores), so their scores cannot be averaged. RRF throws
the scores away and fuses *ranks* — a chunk's contribution is
weight * 1/(k + rank). Chunks that several queries or both legs rank highly
accumulate score; one-list flukes do not.

Reranking: fusion decides what makes the shortlist, but it never reads the
question against a chunk. A cross-encoder does exactly that — too slow to
run over the whole collection, ideal over ~20 survivors.

Everything runs locally: sentence-transformers for embeddings and
reranking, ChromaDB on disk, rank_bm25 in memory, and llama3 through
Ollama's OpenAI-compatible endpoint.
"""

import logging
import os
import re

# Must be set before chromadb is imported to fully suppress telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger("app.rag")

COLLECTION_NAME = "college_notes"


# ══════════════════════════════════════════════════════════════════════
# 1. EMBEDDINGS
#
# An embedding is a vector that encodes meaning: `all-MiniLM-L6-v2` maps
# any text to 384 dimensions where semantically related texts point in
# similar directions. That is what lets a question match a paragraph that
# explains the same idea in different words.
#
# Vectors are normalized to unit length so cosine similarity (the angle
# between them) is the natural comparison and text length does not skew it.
# ══════════════════════════════════════════════════════════════════════

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _get_embedding_model().encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


# ══════════════════════════════════════════════════════════════════════
# 2. VECTOR STORE (dense retrieval leg)
#
# ChromaDB indexes the chunk vectors (HNSW) so nearest-neighbour search
# stays fast as the collection grows, persists to disk so uploads survive
# restarts, and filters on metadata so a question can be scoped to one PDF.
# ══════════════════════════════════════════════════════════════════════

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """Store chunk vectors, text, and citation metadata."""
    if not chunks:
        return
    get_collection().add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"document_id": c["document_id"], "source": c["source"], "page": c["page"]}
            for c in chunks
        ],
    )
    _invalidate_bm25()
    logger.info(f"[EMBED] {len(chunks)} embeddings")


def delete_document(document_id: str) -> None:
    """Drop every chunk belonging to one document."""
    get_collection().delete(where={"document_id": document_id})
    _invalidate_bm25()


def vector_search(
    query: str, top_k: int, document_id: str | None = None
) -> list[dict]:
    """
    Dense leg: nearest neighbours by meaning.

    ChromaDB returns cosine *distance* (0 = identical); we report
    `score = 1 - distance` so higher always means better, and drop anything
    under RETRIEVAL_THRESHOLD — vector search always returns its k nearest
    neighbours even when none are actually relevant.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, count),
        where={"document_id": document_id} if document_id else None,
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    hits = []
    for i, chunk_id in enumerate(results["ids"][0]):
        score = round(1 - results["distances"][0][i], 4)
        if score < settings.RETRIEVAL_THRESHOLD:
            continue
        meta = results["metadatas"][0][i]
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "source": meta["source"],
                "page": meta["page"],
                "score": score,
            }
        )
    return hits


# ══════════════════════════════════════════════════════════════════════
# 3. BM25 (sparse / keyword retrieval leg)
#
# BM25 ranks by term overlap, weighting rare words heavily and saturating
# repeated ones. It cannot understand paraphrase — and that is precisely
# why it complements the dense leg: a question containing "RAID 5",
# "TCP", or "B+ tree" hits the chunk with those literal tokens even when
# the embedding puts a fuzzier paragraph closer.
#
# The index is rebuilt in memory from ChromaDB on first use and thrown
# away whenever documents change. At personal-notes scale that costs
# milliseconds and avoids a second thing on disk to keep in sync.
# ══════════════════════════════════════════════════════════════════════

_bm25: BM25Okapi | None = None
_bm25_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _invalidate_bm25() -> None:
    global _bm25, _bm25_corpus
    _bm25 = None
    _bm25_corpus = []


def _get_bm25() -> BM25Okapi | None:
    global _bm25, _bm25_corpus
    if _bm25 is None:
        collection = get_collection()
        if collection.count() == 0:
            return None

        stored = collection.get(include=["documents", "metadatas"])
        _bm25_corpus = [
            {
                "chunk_id": chunk_id,
                "text": stored["documents"][i],
                "source": stored["metadatas"][i]["source"],
                "page": stored["metadatas"][i]["page"],
                "document_id": stored["metadatas"][i]["document_id"],
            }
            for i, chunk_id in enumerate(stored["ids"])
        ]
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _bm25_corpus])
        logger.info(f"[BM25] Indexed {len(_bm25_corpus)} chunks")
    return _bm25


def bm25_search(query: str, top_k: int, document_id: str | None = None) -> list[dict]:
    """Sparse leg: best term-overlap matches, best first."""
    bm25 = _get_bm25()
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(
        (
            (score, chunk)
            for score, chunk in zip(scores, _bm25_corpus)
            if score > 0
            and (document_id is None or chunk["document_id"] == document_id)
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": round(float(score), 4),
        }
        for score, chunk in ranked[:top_k]
    ]


# ══════════════════════════════════════════════════════════════════════
# 4. MULTI-QUERY EXPANSION
#
# llama3 rewrites the question a few different ways so retrieval gets
# several shots at matching the notes' actual vocabulary. The rewrites are
# a retrieval aid only — the original question is what gets answered, and
# what the reranker scores against.
# ══════════════════════════════════════════════════════════════════════

QUERY_REWRITE_PROMPT = """You rewrite study questions so a search engine can find the right lecture-note passages.

Write {n} alternative phrasings of the question below. Vary the wording and include likely textbook terminology.

Rules:
- Output ONLY the {n} rewritten questions, one per line.
- No numbering, no bullets, no commentary, no blank lines.

Question: {question}"""


def expand_query(question: str, n: int) -> list[str]:
    """
    Return up to `n` alternative phrasings (never including the original).

    A local model will occasionally ignore the format instructions, so the
    output is parsed defensively and any failure degrades to "no variations"
    rather than breaking retrieval.
    """
    try:
        response = _llm_client().chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": QUERY_REWRITE_PROMPT.format(n=n, question=question),
                }
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[MULTIQUERY] Expansion failed, using original query only: {e}")
        return []

    variations: list[str] = []
    seen = {question.strip().lower()}
    for line in raw.splitlines():
        # Strip list markers llama3 adds despite being told not to.
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        # Drop preamble ("Here are 3 alternative phrasings:") — a line
        # ending in a colon is never one of the rewrites.
        if not line or line.endswith(":") or line.lower() in seen:
            continue
        seen.add(line.lower())
        variations.append(line)

    return variations[:n]


# ══════════════════════════════════════════════════════════════════════
# 5. RECIPROCAL RANK FUSION
#
# Each ranked list votes for a chunk with weight * 1/(RRF_K + rank). Only
# positions matter, so lists with incomparable scoring scales (cosine vs
# BM25) fuse cleanly, and a chunk ranked decently by several lists beats
# one ranked first by a single list.
# ══════════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(weighted_lists: list[tuple[list[dict], float]]) -> list[dict]:
    """Fuse (ranked_list, weight) pairs into one ranking, best first."""
    fused: dict[str, dict] = {}

    for ranked_list, weight in weighted_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            contribution = weight / (settings.RRF_K + rank)
            existing = fused.get(chunk["chunk_id"])
            if existing is None:
                fused[chunk["chunk_id"]] = {**chunk, "score": contribution}
            else:
                existing["score"] += contribution

    results = sorted(fused.values(), key=lambda c: c["score"], reverse=True)
    for chunk in results:
        chunk["score"] = round(chunk["score"], 6)
    return results


# ══════════════════════════════════════════════════════════════════════
# 6. RERANKING
#
# Bi-encoders (retrieval) embed the question and the chunk separately, so
# they compare two summaries rather than reading one against the other. A
# cross-encoder takes the pair as a single input and scores "does this
# chunk answer this question" directly — far more accurate, far too slow
# for the whole collection, perfect for a shortlist.
#
# The model is loaded on first use, so ENABLE_RERANKER=false never
# downloads it.
# ══════════════════════════════════════════════════════════════════════

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(settings.RERANKER_MODEL)
    return _cross_encoder


def rerank(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Re-score chunks against the question and keep the best `top_n`."""
    if not chunks:
        return chunks

    scores = _get_cross_encoder().predict([[question, c["text"]] for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk["score"] = round(float(score), 4)

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_n]


# ══════════════════════════════════════════════════════════════════════
# 7. RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════════════════════


def retrieve(
    question: str, top_k: int | None = None, document_id: str | None = None
) -> list[dict]:
    """
    Run the full retrieval pipeline and return the best `top_k` chunks,
    each as {chunk_id, text, source, page, score}, best first.

    Every query variation contributes its own dense list and its own BM25
    list, and all of them go into a single RRF pass — so multi-query
    agreement and dense/sparse agreement are resolved in one step rather
    than fusing twice.
    """
    top_k = top_k or settings.TOP_K
    candidates = max(settings.CANDIDATES, top_k)

    queries = [question]
    if settings.ENABLE_MULTI_QUERY:
        queries += expand_query(question, settings.MULTI_QUERY_COUNT)
        logger.info(f"[MULTIQUERY] {len(queries)} queries: {queries}")

    weighted_lists: list[tuple[list[dict], float]] = []
    for query in queries:
        weighted_lists.append(
            (vector_search(query, candidates, document_id), settings.VECTOR_WEIGHT)
        )
        if settings.ENABLE_HYBRID:
            weighted_lists.append(
                (bm25_search(query, candidates, document_id), settings.BM25_WEIGHT)
            )

    fused = reciprocal_rank_fusion(weighted_lists)[:candidates]
    logger.info(f"[FUSE] {len(fused)} unique chunks from {len(weighted_lists)} lists")

    if not fused:
        return []

    if settings.ENABLE_RERANKER:
        return rerank(question, fused, top_n=top_k)
    return fused[:top_k]


# ══════════════════════════════════════════════════════════════════════
# 8. GENERATION
#
# What makes this RAG rather than "ask an LLM": the prompt carries the
# retrieved chunks, and the system prompt forbids outside knowledge. The
# model explains the evidence; it does not supply it.
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a college study assistant.

Answer the user's question using ONLY the provided context from the uploaded college notes.

Do not use outside knowledge.

If the answer cannot be found in the provided context, say:
"I couldn't find the answer in the uploaded notes."

Do not invent facts.

Explain concepts clearly and concisely.

Always identify the sources used for the answer."""


class LLMError(Exception):
    """Raised when the local Ollama server can't be reached or errors out."""


def _llm_client() -> OpenAI:
    # Ollama ignores the API key, but the SDK requires a non-empty string.
    return OpenAI(api_key="ollama", base_url=settings.LLM_BASE_URL)


def build_context(chunks: list[dict]) -> str:
    """Render chunks as source-labelled blocks so the model (and anyone
    reading the prompt while debugging) can see where each fact came from."""
    return "\n\n".join(
        f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Answer `question` grounded in `chunks`. Raises LLMError on failure."""
    user_message = f"CONTEXT\n\n{build_context(chunks)}\n\nQUESTION\n\n{question}"

    try:
        response = _llm_client().chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
    except Exception as e:
        raise LLMError(
            f"Could not reach Ollama at {settings.LLM_BASE_URL} with model "
            f"'{settings.LLM_MODEL}'. Is Ollama running (`ollama serve`) and "
            f"have you pulled this model (`ollama pull {settings.LLM_MODEL}`)? "
            f"Original error: {e}"
        )

    return response.choices[0].message.content
