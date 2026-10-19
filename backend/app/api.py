from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from langchain_core.documents import Document

from app import ingest, rag
from app.rag import LLMError
from app.schemas import ChatRequest, ChatResponse, DeleteResponse, DocumentOut, RetrievedChunkOut, SourceOut, UploadResponse


documents_router = APIRouter(prefix="/api/documents", tags=["documents"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
NOT_FOUND_ANSWER = "I couldn't find the answer in the uploaded notes."


@documents_router.get("", response_model=list[DocumentOut])
def get_documents():
    return ingest.list_documents()


@documents_router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 50 MB limit.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File does not look like a valid PDF.")

    record = ingest.save_document(file.filename, content)
    background_tasks.add_task(ingest.process_document, record["document_id"])
    return UploadResponse(document=DocumentOut(**record), message=f"'{record['filename']}' uploaded; indexing started.")


@documents_router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    if not ingest.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, message="Document deleted.")


def _dedupe_sources(docs: list[Document]) -> list[SourceOut]:
    best: dict = {}
    for d in docs:
        key = (d.metadata["source"], d.metadata["page"])
        score = d.metadata.get("rerank_score")
        if key not in best or (score is not None and (best[key] is None or score > best[key])):
            best[key] = score
    return [SourceOut(document=name, page=page, score=score) for (name, page), score in sorted(best.items())]


@chat_router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        docs, trace = rag.retrieve_with_trace(question, document_id=request.document_id)
        if not docs:
            return ChatResponse(
                answer=NOT_FOUND_ANSWER,
                sources=[],
                trace=trace if request.include_trace else None,
            )
        answer = rag.generate_answer(question, docs)
        trace["final_context"] = rag.format_docs(docs)
        return ChatResponse(
            answer=answer,
            sources=_dedupe_sources(docs),
            trace=trace if request.include_trace else None,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@chat_router.post("/debug-retrieve", response_model=list[RetrievedChunkOut])
def debug_retrieve(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        docs = rag.retrieve(question, document_id=request.document_id)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [
        RetrievedChunkOut(
            text=d.page_content,
            source=d.metadata["source"],
            page=d.metadata["page"],
            chunk_id=d.metadata["chunk_id"],
            score=d.metadata.get("rerank_score"),
        )
        for d in docs
    ]