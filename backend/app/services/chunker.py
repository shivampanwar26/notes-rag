"""
Chunking (Phase 2).

CONCEPT — what is a chunk, and why do we need chunking at all?
Embedding models and LLM context windows both have practical size limits,
and more importantly: embedding a *whole 80-page PDF* into a single vector
would blur together dozens of unrelated topics (paging, scheduling,
deadlocks...) into one mushy average. That vector wouldn't be close to
any specific question. A "chunk" is a small, semantically coherent slice
of text (a paragraph or a few sentences) that gets its own embedding, so
a search for "demand paging" can match the one paragraph about demand
paging instead of the entire textbook.

CONCEPT — why does chunk size matter?
  - Too large: a chunk mixes multiple topics, so its embedding is a vague
    average that doesn't strongly match any single question. You also
    waste LLM context on irrelevant text sitting next to the relevant part.
  - Too small: a chunk loses surrounding context (e.g. just one sentence
    of a multi-sentence explanation), so the LLM gets a fragment that's
    hard to answer from, and you multiply the number of vectors to store
    and search.
  700 characters (~120-150 words) is a common, reasonable middle ground
  for dense textbook-style prose — configurable via CHUNK_SIZE.

CONCEPT — why overlap between chunks?
If a concept's explanation happens to fall right on a chunk boundary,
splitting it exactly in half means neither chunk fully contains the idea.
Overlap (CHUNK_OVERLAP characters repeated at the start of the next chunk)
means boundary-straddling ideas still appear whole in at least one chunk.

CONCEPT — why "sensible" chunking instead of a fixed character cut?
Cutting exactly every N characters can slice a sentence in half mid-word.
We use LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph breaks first, then sentence breaks, then words — only falling
back to a hard character cut as a last resort. This is one of the few
places LangChain genuinely simplifies things: reimplementing "try these
separators in priority order, recursively" ourselves would be a lot of
fiddly code for something LangChain already does well.

Chunking happens PER PAGE (not on the whole document glued together) so
that every resulting chunk can be tagged with the exact page it came from
— this is what makes accurate citations possible in Phase 6.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger("app.chunker")


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(
    document_id: str, filename: str, pages: list[tuple[int, str]]
) -> list[dict]:
    """
    Turn (page_number, text) pairs into a flat list of chunk dicts, each
    carrying the metadata that later phases (embeddings, vector store,
    citations) depend on:

        {
            "chunk_id":    "<document_id>-p<page>-c<index>",
            "text":        "...",
            "document_id": "...",
            "source":      "Operating Systems.pdf",
            "page":        42,
        }
    """
    splitter = _make_splitter()
    chunks: list[dict] = []

    for page_number, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue  # blank pages (cover pages, section dividers) produce no chunks

        pieces = splitter.split_text(page_text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{document_id}-p{page_number}-c{i}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "document_id": document_id,
                    "source": filename,
                    "page": page_number,
                }
            )

    logger.info(f"[CHUNK] {len(chunks)} chunks")
    return chunks
