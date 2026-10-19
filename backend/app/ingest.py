from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import rag

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

_document_status: dict[str, str] = {}
_document_errors: dict[str, str] = {}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


class PDFProcessingError(Exception):
    pass


def load_chunks(document_id: str, path: Path, filename: str):
    try:
        pages = PyMuPDFLoader(str(path)).load()
    except Exception as exc:
        raise PDFProcessingError(f"Could not read PDF: {exc}") from exc

    if not any(p.page_content.strip() for p in pages):
        raise PDFProcessingError("No extractable text found. Scanned PDFs need OCR.")

    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata = {
            "document_id": document_id,
            "chunk_id": str(uuid4()),
            "source": filename,
            "page": int(chunk.metadata.get("page", 0)) + 1,
        }
    return chunks


def document_path(document_id: str) -> Path:
    return next(DOCUMENTS_DIR.glob(f"{document_id}__*.pdf"), DOCUMENTS_DIR / f"{document_id}.pdf")

def save_document(filename: str, content: bytes) -> dict:
    filename = Path(filename.replace("\\", "/")).name

    if not filename or filename in {".", ".."}:
        raise PDFProcessingError("Invalid filename.")

    if not content:
        raise PDFProcessingError("The uploaded file is empty.")

    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Only PDF files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid4())
    path = DOCUMENTS_DIR / f"{document_id}__{filename}"
    path.write_bytes(content)

    _document_status[document_id] = "processing"
    _document_errors.pop(document_id, None)

    return {
        "document_id": document_id,
        "filename": filename,
        "size_bytes": len(content),
        "uploaded_at": datetime.now(timezone.utc),
        "status": "processing",
        "page_count": None,
        "chunk_count": None,
        "error": None,
    }


def _stored_document_info() -> dict:
    info: dict = {}
    for meta in rag.get_vectorstore().get(include=["metadatas"])["metadatas"] or []:
        if meta and (doc_id := meta.get("document_id")):
            item = info.setdefault(doc_id, {"pages": set(), "chunks": 0})
            item["chunks"] += 1
            item["pages"].add(meta.get("page"))
    return info


def list_documents() -> list[dict]:
    stored = _stored_document_info()
    documents = []
    for path in DOCUMENTS_DIR.glob("*.pdf"):
        document_id, _, filename = path.name.partition("__")
        info = stored.get(document_id, {})
        stat = path.stat()
        documents.append({
            "document_id": document_id,
            "filename": filename or path.name,
            "size_bytes": stat.st_size,
            "uploaded_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            "status": _document_status.get(document_id, "indexed" if info else "uploaded"),
            "page_count": len(info.get("pages", ())) or None,
            "chunk_count": info.get("chunks"),
            "error": _document_errors.get(document_id),
        })
    return sorted(documents, key=lambda d: d["uploaded_at"], reverse=True)


def get_document(document_id: str) -> dict | None:
    return next((d for d in list_documents() if d["document_id"] == document_id), None)


def delete_document(document_id: str) -> bool:
    path = document_path(document_id)
    if not path.exists():
        return False
    rag.delete_document(document_id)
    path.unlink()
    _document_status.pop(document_id, None)
    _document_errors.pop(document_id, None)
    return True


def process_document(document_id: str) -> None:
    path = document_path(document_id)
    if not path.exists():
        return
    try:
        filename = path.name.partition("__")[2] or path.name
        chunks = load_chunks(document_id, path, filename)
        if not chunks:
            raise PDFProcessingError("No text chunks were produced.")
        rag.add_documents(chunks)  # embeds + stores in Chroma
        _document_status[document_id] = "indexed"
        _document_errors.pop(document_id, None)
    except Exception as exc:
        _document_status[document_id] = "failed"
        _document_errors[document_id] = str(exc)