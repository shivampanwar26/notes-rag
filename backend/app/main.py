from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router, documents_router

app = FastAPI(
    title="College Notes RAG API",
    description="Upload college notes and ask questions with Gemini-powered RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
