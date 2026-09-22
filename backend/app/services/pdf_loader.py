"""
PDF text extraction (Phase 2).

CONCEPT — why page-by-page, not "whole document as one string"?
Citations are the whole point of a trustworthy study assistant: when the
app says "demand paging is explained on page 42," that claim has to be
traceable back to a real page. If we extracted the PDF as one long blob
of text, we'd lose page boundaries and could never cite accurately again.
So the very first thing we do is extract text per page and keep the page
number attached to it — every downstream step (chunking, embedding,
storage) carries that page number forward.

We use PyMuPDF (`fitz`) because it's fast, has no external system
dependencies (unlike some OCR-based tools), and gives clean plain text
for text-based PDFs, which is what lecture notes/slides almost always are.
"""

import logging

import fitz  # PyMuPDF

logger = logging.getLogger("app.pdf_loader")


class PDFProcessingError(Exception):
    """Raised for any PDF that can't be turned into usable text."""


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    """
    Extract text from every page of a PDF.

    Returns a list of (page_number, text) tuples, 1-indexed to match how
    humans refer to PDF pages ("see page 42"), not how programmers index
    arrays.

    Raises PDFProcessingError for:
      - a file that isn't actually a valid/openable PDF (corrupted)
      - a PDF with zero pages
      - a PDF where every page has no extractable text (e.g. a pure
        image/scan with no OCR text layer) — we can't ground answers in
        text we can't read, so we fail loudly instead of silently
        indexing nothing.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFProcessingError(f"Could not open PDF (file may be corrupted): {e}")

    if doc.page_count == 0:
        doc.close()
        raise PDFProcessingError("PDF has no pages.")

    pages: list[tuple[int, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append((i + 1, text))
    doc.close()

    total_chars = sum(len(t.strip()) for _, t in pages)
    if total_chars == 0:
        raise PDFProcessingError(
            "No extractable text found. This PDF may be a scanned image "
            "with no text layer — OCR is not supported yet."
        )

    logger.info(f"[PARSE] {len(pages)} pages")
    return pages
