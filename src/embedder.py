"""Thin wrapper around a multilingual sentence-embedding model."""
from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def _is_e5() -> bool:
    return "e5" in config.EMBED_MODEL.lower()


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    prefix = "passage: " if _is_e5() else ""
    return _model().encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    ).astype("float32")


def embed_queries(texts: list[str]) -> np.ndarray:
    prefix = "query: " if _is_e5() else ""
    return _model().encode(
        [prefix + t for t in texts], normalize_embeddings=True
    ).astype("float32")
