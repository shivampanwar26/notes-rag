import os
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
COLLECTION_NAME = "college_notes"

TOP_K = 5
CANDIDATES = 20
RETRIEVAL_THRESHOLD = 0.3
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
ENABLE_HYBRID = True
ENABLE_MULTI_QUERY = True
MULTI_QUERY_COUNT = 3
ENABLE_RERANKER = True


class LLMError(Exception):
    pass


# ---------- Lazy singletons ----------
_state: dict = {}
_bm25_cache: dict = {}


def _once(key, factory):
    if key not in _state:
        _state[key] = factory()
    return _state[key]


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is missing. Add it to backend/.env and restart the backend.")
    return _once(
        ("llm", temperature),
        lambda: ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=temperature),
    )


def get_vectorstore() -> Chroma:
    return _once("vs", lambda: Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        ),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    ))


def get_cross_encoder() -> HuggingFaceCrossEncoder:
    return _once("cross_encoder", lambda: HuggingFaceCrossEncoder(model_name=RERANKER_MODEL))


# ---------- Indexing ----------
def add_documents(chunks: list[Document]) -> None:
    if chunks:
        get_vectorstore().add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        _bm25_cache.clear()


def delete_document(document_id: str) -> None:
    vs = get_vectorstore()
    ids = vs.get(where={"document_id": document_id})["ids"]
    if ids:
        vs.delete(ids=ids)
    _bm25_cache.clear()


# ---------- Retrieval ----------
def _bm25_retriever(document_id: str | None):
    if document_id not in _bm25_cache:
        data = get_vectorstore().get(
            where={"document_id": document_id} if document_id else None,
            include=["documents", "metadatas"],
        )
        docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
        _bm25_cache[document_id] = BM25Retriever.from_documents(docs, k=CANDIDATES) if docs else None
    return _bm25_cache[document_id]


MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    """You help search college lecture notes. Rewrite the question into {count} alternative search queries.
Vary wording and include likely textbook terms. Return only one query per line, with no numbering or explanation.

Question: {question}"""
)


def build_retriever(document_id: str | None = None):
    """Vector retriever, fused with BM25 via weighted RRF when hybrid search is on."""
    vector = get_vectorstore().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": CANDIDATES,
            "score_threshold": RETRIEVAL_THRESHOLD,
            **({"filter": {"document_id": document_id}} if document_id else {}),
        },
    )
    bm25 = _bm25_retriever(document_id) if ENABLE_HYBRID else None
    if not bm25:
        return vector
    return EnsembleRetriever(retrievers=[vector, bm25], weights=[VECTOR_WEIGHT, BM25_WEIGHT])


def expand_query(question: str) -> list[str]:
    chain = MULTI_QUERY_PROMPT | get_llm(0.3) | LineListOutputParser()
    seen = {question.strip().lower()}
    variations = []
    for q in chain.invoke({"question": question, "count": MULTI_QUERY_COUNT}):
        q = q.strip().strip('"')
        if q and q.lower() not in seen:
            seen.add(q.lower())
            variations.append(q)
    return variations[:MULTI_QUERY_COUNT]


def _brief(doc: Document, **extra) -> dict:
    return {
        "chunk_id": doc.metadata.get("chunk_id"),
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "text": doc.page_content,
        **extra,
    }


def retrieve_with_trace(question: str, document_id: str | None = None) -> tuple[list[Document], dict]:
    started = time.perf_counter()

    # 1. query expansion
    queries, expansion_error = [question], None
    if ENABLE_MULTI_QUERY:
        try:
            queries += expand_query(question)
        except Exception as exc:
            expansion_error = str(exc)

    # 2. hybrid retrieval per query, then union (deduplicated by chunk)
    retriever = build_retriever(document_id)
    search_trace, union = [], {}
    for i, query in enumerate(queries):
        docs = retriever.invoke(query)
        search_trace.append({
            "query": query,
            "query_label": "Original query" if i == 0 else f"Alternative query {i}",
            "results": [_brief(d) for d in docs],
        })
        for d in docs:
            union.setdefault(d.metadata["chunk_id"], d)
    candidates = list(union.values())

    # 3. cross-encoder reranking
    if ENABLE_RERANKER and candidates:
        scores = get_cross_encoder().score([(question, d.page_content) for d in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:TOP_K]
        final = [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": round(float(s), 4)})
            for d, s in ranked
        ]
    else:
        final = candidates[:TOP_K]

    trace = {
        "original_query": question,
        "document_id": document_id,
        "queries": queries,
        "expansion_error": expansion_error,
        "search_results": search_trace,
        "before_reranking": [_brief(d) for d in candidates],
        "after_reranking": [_brief(d, rerank_score=d.metadata.get("rerank_score")) for d in final],
        "settings": {
            "hybrid_search": ENABLE_HYBRID,
            "multi_query": ENABLE_MULTI_QUERY,
            "reranker": ENABLE_RERANKER,
            "vector_weight": VECTOR_WEIGHT,
            "bm25_weight": BM25_WEIGHT if ENABLE_HYBRID else 0,
            "top_k": TOP_K,
            "candidate_limit": CANDIDATES,
        },
        "retrieval_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    return final, trace


def retrieve(question: str, document_id: str | None = None) -> list[Document]:
    return retrieve_with_trace(question, document_id)[0]


# ---------- Generation ----------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a college study assistant. Answer using only the provided context from the uploaded "
     "college notes. Do not use outside knowledge or invent facts. If the answer is not in the "
     "context, say: \"I couldn't find the answer in the uploaded notes.\" Explain clearly and "
     "concisely. Cite relevant source filenames and page numbers when possible."),
    ("human", "CONTEXT FROM UPLOADED NOTES:\n{context}\n\nQUESTION:\n{question}\n\nAnswer only from the context above."),
])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source']} | Page {d.metadata['page']} | Chunk: {d.metadata['chunk_id']}]\n{d.page_content}"
        for d in docs
    )


def generate_answer(question: str, docs: list[Document]) -> str:
    chain = ANSWER_PROMPT | get_llm(0.2) | StrOutputParser()
    try:
        return chain.invoke({"context": format_docs(docs), "question": question}).strip()
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Gemini request failed using model '{GEMINI_MODEL}': {exc}") from exc


def answer_question(question: str, document_id: str | None = None, include_trace: bool = False) -> dict:
    """One-call helper the chat router can use; returns a dict matching ChatResponse."""
    docs, trace = retrieve_with_trace(question, document_id)
    answer = generate_answer(question, docs) if docs else "I couldn't find the answer in the uploaded notes."
    return {
        "answer": answer,
        "sources": [{"document": d.metadata["source"], "page": d.metadata["page"], "score": d.metadata.get("rerank_score")} for d in docs],
        "trace": trace if include_trace else None,
    }