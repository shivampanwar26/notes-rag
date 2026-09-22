"""
Embeddings (Phase 3).

CONCEPT — what is an embedding?
An embedding is a fixed-length list of numbers (a vector) that represents
the *meaning* of a piece of text. `all-MiniLM-L6-v2` turns any text into
a 384-dimensional vector. Two pieces of text that mean similar things
end up as two vectors that point in similar directions in that
384-dimensional space — even if they don't share any of the same words.

CONCEPT — why does semantically similar text produce similar vectors?
The model was trained on millions of sentence pairs to pull semantically
related sentences' vectors close together and push unrelated ones apart.
That's what makes it possible to match a question like "What is demand
paging?" against a chunk that says "...pages are loaded into memory only
when referenced..." — no shared keywords, but related meaning, so their
vectors end up close together.

CONCEPT — cosine similarity.
Cosine similarity measures the angle between two vectors, ignoring their
length: 1.0 means pointing in exactly the same direction (same meaning),
0 means unrelated, -1 means opposite. We use it (instead of e.g. raw
Euclidean distance) because embedding *direction* carries the meaning;
two vectors can point the same way but have different magnitudes just
from text length, which we don't want to penalize.

Implementation notes:
  - We load the model once (singleton) — loading is the slow part
    (downloading + initializing weights), embedding calls after that are fast.
  - `normalize_embeddings=True` makes every vector unit length, which is
    what lets ChromaDB's cosine-space search work cleanly and is the
    standard practice for this model.
  - The model name comes from settings.EMBEDDING_MODEL (.env), so
    swapping to a different Sentence-Transformers model later means
    changing one line in .env, not this file.
"""

import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger("app.embeddings")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts (used during indexing)."""
    model = get_model()
    vectors = model.encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single user question (used during retrieval)."""
    return embed_texts([text])[0]
