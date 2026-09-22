"""
Centralized configuration for the whole backend.

Every setting the app needs (paths, chunking parameters, model names,
retrieval knobs) is read from environment variables here ONCE and imported
everywhere else as `settings`, so changing behaviour is a `.env` edit
rather than a code edit.

The LLM is always a local Ollama model (llama3 by default) reached through
Ollama's OpenAI-compatible endpoint — no API key, no hosted provider, no
data leaving the machine.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Paths ---------------------------------------------------------
    # BASE_DIR = backend/  (this file lives in backend/app/config.py)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    CHROMA_DIR: Path = DATA_DIR / "chroma"

    # ---- CORS ----------------------------------------------------------
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    # ---- Ingestion -----------------------------------------------------
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Retrieval -----------------------------------------------------
    TOP_K: int = 5
    # Minimum cosine similarity a chunk must reach on the dense (vector)
    # leg to enter fusion. BM25 results are never threshold-filtered: the
    # whole point of the keyword leg is to surface exact terms/acronyms
    # that embeddings score poorly.
    RETRIEVAL_THRESHOLD: float = 0.3
    # Candidates pulled per ranked list before fusion + reranking narrow
    # them down to TOP_K.
    CANDIDATES: int = 20

    # ---- Hybrid search (dense + BM25, fused with RRF) -------------------
    ENABLE_HYBRID: bool = True
    # Relative pull of each leg inside RRF. Dense is weighted higher because
    # notes are prose (meaning matters more than literal wording), with BM25
    # kept as a meaningful minority vote for exact terms.
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # ---- Multi-query expansion -----------------------------------------
    ENABLE_MULTI_QUERY: bool = True
    MULTI_QUERY_COUNT: int = 3

    # ---- Reciprocal Rank Fusion ----------------------------------------
    # The constant in 1/(RRF_K + rank). Larger = gentler penalty for lower
    # ranks; 60 is the value from the original RRF paper.
    RRF_K: int = 60

    # ---- Reranking (cross-encoder, final precision stage) ---------------
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- LLM (local Ollama only) ---------------------------------------
    LLM_MODEL: str = "llama3"
    LLM_BASE_URL: str = "http://localhost:11434/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
