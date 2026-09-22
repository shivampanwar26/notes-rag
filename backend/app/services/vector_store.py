"""
Vector store (Phase 3).

CONCEPT — why a vector database instead of, say, a Python list?
Once you have thousands of chunk embeddings, "which of these is closest
to my question's embedding?" becomes a nearest-neighbor search problem.
A naive linear scan (compare the query against every vector one by one)
works but gets slow as the collection grows. Vector databases like
ChromaDB build an index (ChromaDB uses HNSW — Hierarchical Navigable
Small World graphs) that makes this search fast even at scale, while
also handling persistence and metadata filtering for you.

We chose ChromaDB specifically because it runs embedded/local (no server
to stand up, no account, no network calls), persists to a folder on disk,
and has a simple Python API — ideal for a personal notes app and for
learning without extra infrastructure.

This file is the ONLY place that talks to ChromaDB. Everything else
(retriever.py, document_processor.py) calls these functions.
"""

import logging
import os

# Must be set before chromadb is imported to fully suppress its telemetry
# (the anonymized_telemetry=False client setting below still lets it try
# and log harmless "failed to send" errors on some versions).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb

from app.config import settings

logger = logging.getLogger("app.vector_store")

COLLECTION_NAME = "college_notes"

_client = None
_collection = None


def get_collection():
    """
    Lazily create a persistent ChromaDB client + collection.

    `hnsw:space: "cosine"` tells ChromaDB to rank results by cosine
    similarity (matching how the embedding model was designed to be
    compared) instead of its default (squared L2 distance).

    Persistence: PersistentClient writes to settings.CHROMA_DIR, so the
    index survives backend restarts — you don't need to re-upload PDFs
    every time you restart the server.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Store chunk vectors + text + metadata in ChromaDB.

    Each chunk dict is expected to have: chunk_id, text, document_id,
    source, page (see chunker.py). We store `document_id` and `page` as
    metadata so we can later filter by document (Phase 4) and cite pages
    (Phase 6) without re-parsing anything.
    """
    if not chunks:
        return
    collection = get_collection()
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "document_id": c["document_id"],
                "source": c["source"],
                "page": c["page"],
            }
            for c in chunks
        ],
    )
    logger.info(f"[EMBED] {len(chunks)} embeddings")


def search(
    query_embedding: list[float], top_k: int, document_id: str | None = None
) -> list[dict]:
    """
    Find the top_k chunks whose embeddings are closest to query_embedding.

    Returns a list of dicts: {chunk_id, text, source, page, score}.
    `score` is cosine similarity in [0, 1] — we convert ChromaDB's raw
    cosine *distance* (0 = identical, 2 = opposite) into similarity
    (1 = identical) as `score = 1 - distance`, since "higher = better
    match" is far more intuitive than "lower = better match".
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    where = {"document_id": document_id} if document_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    out = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        out.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "score": round(1 - distance, 4),
            }
        )
    return out


def delete_document(document_id: str) -> None:
    """Remove every chunk vector belonging to one document (on document delete/re-upload)."""
    collection = get_collection()
    collection.delete(where={"document_id": document_id})


def is_indexed(document_id: str) -> bool:
    """Whether any chunks for this document are already in the vector store."""
    collection = get_collection()
    result = collection.get(where={"document_id": document_id}, limit=1)
    return len(result["ids"]) > 0
