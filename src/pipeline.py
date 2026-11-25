"""End-to-end question answering: retrieve -> generate. Shared by the API and the UI."""
from functools import lru_cache

from . import generate
from .retrieve import Retriever


@lru_cache(maxsize=4)
def get_retriever(chunking: str) -> Retriever:
    return Retriever(chunking)


def ask(question: str, mode: str = "hybrid_rerank", chunking: str = "semantic",
        k: int = 5, strict: bool = True) -> dict:
    chunks = get_retriever(chunking).search(question, mode=mode, k=k)
    result = generate.answer(question, chunks, strict=strict)
    result["sources"] = chunks
    return result
