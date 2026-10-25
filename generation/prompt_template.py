"""The grounding-only prompt template: instructs the LLM to answer using
ONLY the retrieved context (from retrieval/hybrid_retrieve.py) and to say
so explicitly when the context doesn't support an answer, rather than
falling back on the model's own training-time knowledge. This is the core
anti-hallucination lever for Part 5 -- see docs/part5_notes.md.
"""

REFUSAL_PHRASE = "The provided context does not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You are a historical question-answering assistant for Indian history.

Answer ONLY using the context provided below the question. Do not use any outside knowledge, even if you are confident it is correct -- the context is the only source of truth you are allowed to draw on.

The context is a numbered list of sources: facts from a knowledge graph and passages from a book.

Rules:
1. If the context fully answers the question, answer concisely. After each sentence, cite the sources that support it by their numbers, e.g. [2] or [1][4]. Use only numbers that appear in the context, and cite only with these numbers -- never write file names, page numbers or source titles.
2. If the context only partially answers the question, answer what you can from it and clearly state what is missing.
3. If the context does not contain information relevant to the question, respond exactly: "{REFUSAL_PHRASE}" Do not guess, speculate, or fill gaps from prior knowledge.
4. Never invent names, dates, places, or events that are not present in the context.
5. "X and Y are mentioned together in the book" only means the book names both near each other. It is not a relationship: do not conclude from it that X led, fought, helped or was allied with Y.
6. Say who did what, or who was on which side, only when a source states it.
7. Write plain sentences: no headings, bullet points or bold text, and do not begin with remarks about the context such as "The provided context contains...".
"""


def build_messages(question: str, context: str) -> list:
    """Build an Ollama /api/chat-style messages list for a grounded answer."""
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


CLOSED_BOOK_REFUSAL = "I do not know."

CLOSED_BOOK_SYSTEM_PROMPT = f"""You are a historical question-answering assistant for Indian history.

Answer the question concisely from your own knowledge. If you are not sure of the answer, reply exactly: "{CLOSED_BOOK_REFUSAL}" Do not guess.
"""


def build_closed_book_messages(question: str) -> list:
    """The no-retrieval baseline: same task, no context."""
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
