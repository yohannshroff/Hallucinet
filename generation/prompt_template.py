"""The grounding-only prompt template: instructs the LLM to answer using
ONLY the retrieved context (from retrieval/hybrid_retrieve.py) and to say
so explicitly when the context doesn't support an answer, rather than
falling back on the model's own training-time knowledge. This is the core
anti-hallucination lever for Week 5 -- see docs/week5_notes.md.
"""

REFUSAL_PHRASE = "The provided context does not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You are a historical question-answering assistant specialized in the Indian Revolt of 1857.

Answer ONLY using the context provided below the question. Do not use any outside knowledge, even if you are confident it is correct -- the context is the only source of truth you are allowed to draw on.

Rules:
1. If the context fully answers the question, answer concisely and mention which source(s) support each claim (use the URLs given in the context).
2. If the context only partially answers the question, answer what you can from it and clearly state what is missing.
3. If the context does not contain information relevant to the question, respond exactly: "{REFUSAL_PHRASE}" Do not guess, speculate, or fill gaps from prior knowledge.
4. Never invent names, dates, places, or events that are not present in the context.
"""


def build_messages(question: str, context: str) -> list:
    """Build an Ollama /api/chat-style messages list for a grounded answer."""
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
