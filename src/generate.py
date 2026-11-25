"""Answer generation with citations and an optional strict 'refuse if unsure' prompt."""
import re

from . import llm

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = "I could not find this in the provided documents."

STRICT_SYSTEM = f"""You answer questions about government policy documents using ONLY the numbered context passages.
Rules:
- Answer in the same language as the question (English or Hindi).
- Cite the passages that support each claim like [1] or [2][3].
- Do not use outside knowledge and do not guess.
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}
- Be concise (at most 4 sentences)."""

LENIENT_SYSTEM = """You are a helpful assistant answering questions about government policy documents.
Use the numbered context passages where possible and cite them like [1]. Answer in the same language as the question."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['doc']}, p.{c['page_start']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def normalize_citations(text: str) -> str:
    """Some models write citations as 【1】 or 【1†source】; turn them into plain [1]."""
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)


def answer(question: str, chunks: list[dict], strict: bool = True) -> dict:
    system = STRICT_SYSTEM if strict else LENIENT_SYSTEM
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    text = llm.chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}])
    text = normalize_citations(text).strip()
    refused = text.upper().startswith(NOT_FOUND_TOKEN)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    return {
        "answer": NOT_FOUND_MESSAGE if refused else text,
        "raw_answer": text,
        "refused": refused,
        "cited": cited,
    }
