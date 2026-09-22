"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
(from inside backend/, with the virtualenv active)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, documents
from app.config import settings

# A simple, readable log format. In later phases we'll log each pipeline
# step ([UPLOAD], [PARSE], [CHUNK], [EMBED], [QUERY], ...) so the RAG
# pipeline is observable while you learn it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="College Notes RAG API",
    description="Upload your college notes and ask questions about them.",
    version="0.1.0",
)

# The React dev server (Vite) runs on a different port than FastAPI,
# so the browser treats them as different origins. CORS middleware tells
# the browser it's safe for the frontend origin to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort safety net (Phase 10): any exception we didn't anticipate
    and turn into a proper HTTPException still returns clean JSON with a
    500 status instead of leaking a raw stack trace to the frontend.
    Specific, expected failures (bad PDF, empty question, LLM error, ...)
    are handled with precise HTTPExceptions closer to where they happen —
    this handler is only a backstop.
    """
    logging.getLogger("app.main").exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
