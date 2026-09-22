"""
LLM generation — the "G" in RAG (Phase 5).

Pipeline:
    Top-K chunks -> context string -> prompt -> LLM -> answer

CONCEPT — "Augmented" generation: what actually makes this RAG and not
just "ask an LLM a question"?
A plain LLM call answers from whatever it memorized during training —
it has never seen your specific PDFs, and it can't tell you which page
something came from. Here, we *augment* the prompt with the actual
retrieved text from your notes before asking the LLM anything. The LLM
never sees your raw PDFs and never uses outside knowledge by instruction
— it only reasons over the handful of chunks retrieval decided were
relevant. That's the "Retrieval-Augmented" part: retrieval supplies the
facts, generation supplies the fluent explanation of those facts.

CONCEPT — why only send the top-K chunks, never the whole PDF?
Two reasons: (1) LLM context windows and inference time both scale with
how much text you send — sending an 80-page PDF on every question is
slow, even running locally; (2) precision — burying the 3 relevant
paragraphs inside 80 pages of unrelated content makes it *more* likely
the model gets distracted or hallucinates a connection that isn't really
there. Precise, targeted context produces more grounded answers.

CONCEPT — the system prompt is what enforces groundedness.
Telling the model "answer using ONLY the provided context" and "if the
answer cannot be found, say so" is the actual mechanism that prevents
the model from quietly filling gaps with outside knowledge. It's not a
technical guarantee (an LLM can still occasionally ignore instructions —
smaller local models are more prone to this than large hosted ones), but
it is the standard, effective way to bias generation toward the provided
evidence.

WHY OLLAMA:
This project runs entirely against a local LLM through Ollama
(https://ollama.com) instead of a paid hosted API — no API key, no data
leaving your machine, no per-token cost. Ollama exposes an
OpenAI-compatible endpoint at /v1, so we can reuse the standard `openai`
Python SDK (just pointed at localhost) instead of writing a bespoke HTTP
client.

SETUP:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3
    3. Make sure Ollama is running (it runs as a background service after
       install; `ollama serve` starts it manually if needed).
    4. In backend/.env, set LLM_MODEL to match the model you pulled.
"""

import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger("app.llm")

SYSTEM_PROMPT = """You are a college study assistant.

Answer the user's question using ONLY the provided context from the uploaded college notes.

Do not use outside knowledge.

If the answer cannot be found in the provided context, say:
"I couldn't find the answer in the uploaded notes."

Do not invent facts.

Explain concepts clearly and concisely.

Always identify the sources used for the answer."""


class LLMError(Exception):
    """Raised when the local Ollama server can't be reached or returns an error."""


def _get_client() -> OpenAI:
    # Ollama's OpenAI-compatible endpoint doesn't check the API key, but
    # the SDK requires some non-empty string to be passed.
    return OpenAI(api_key="ollama", base_url=settings.LLM_BASE_URL)


def build_context(chunks: list[dict]) -> str:
    """
    Render retrieved chunks into the labeled context block the prompt
    expects, e.g.:

        [Source: Operating Systems.pdf | Page 42]
        <chunk text>

        [Source: Operating Systems.pdf | Page 43]
        <chunk text>

    Labeling each chunk with its source is what lets the LLM (and us,
    reading the prompt while debugging) see exactly where each piece of
    context came from — the same metadata that becomes the citations
    in Phase 6.
    """
    blocks = [f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call the local Ollama model with the question grounded in `chunks`. Raises LLMError on failure."""
    context = build_context(chunks)
    user_message = f"CONTEXT\n\n{context}\n\nQUESTION\n\n{question}"

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
    except Exception as e:
        raise LLMError(
            f"Could not reach Ollama at {settings.LLM_BASE_URL} with model "
            f"'{settings.LLM_MODEL}'. Is Ollama running (`ollama serve`) and "
            f"have you pulled this model (`ollama pull {settings.LLM_MODEL}`)? "
            f"Original error: {e}"
        )

    return response.choices[0].message.content
