"""
Centralized configuration for the whole backend.

Every setting the app needs (file paths, chunking parameters, model names,
LLM provider, etc.) is read from environment variables here, ONCE, and
imported everywhere else as `settings`. This means:

- No file ever hard-codes a secret or a magic number.
- Changing behavior (e.g. swapping the embedding model, or moving from
  an API-based LLM to a local Ollama model) is a .env edit, not a code edit.

We only *use* a handful of these settings in Phase 1 (paths, CORS).
The rest (chunking, embeddings, retrieval, LLM) are already defined here
so that later phases just import them instead of re-plumbing config.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Paths -----------------------------------------------------
    # BASE_DIR = backend/  (this file lives in backend/app/config.py)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    CHROMA_DIR: Path = DATA_DIR / "chroma"

    # ---- CORS --------------------------------------------------------
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    # ---- Chunking (used from Phase 2 onward) --------------------------
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100

    # ---- Embeddings (used from Phase 3 onward) ------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Retrieval (used from Phase 4 onward) --------------------------
    TOP_K: int = 5
    # Minimum cosine similarity (0-1) a chunk must have to be considered
    # "relevant". Chunks below this are dropped before ever reaching the
    # LLM — this is what lets us honestly say "not found in your notes"
    # instead of forcing the LLM to answer from a weak/irrelevant chunk.
    RETRIEVAL_THRESHOLD: float = 0.3

    # ---- Reranking (used from Phase 8 onward) ---------------------------
    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many candidates the vector search pulls BEFORE reranking narrows
    # them down to TOP_K. Only used when ENABLE_RERANKER=true.
    RERANK_CANDIDATES: int = 20

    # ---- LLM (used from Phase 5 onward) ---------------------------------
    # This project is configured for Ollama (a local LLM server) only.
    # LLM_BASE_URL points at Ollama's OpenAI-compatible endpoint, and
    # LLM_MODEL must be a model you've already pulled with `ollama pull`.
    LLM_MODEL: str = "llama3"
    LLM_BASE_URL: str = "http://localhost:11434/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Make sure the data directories exist as soon as the app boots.
settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
