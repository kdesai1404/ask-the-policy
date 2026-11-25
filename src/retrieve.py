"""Retrieval: dense, BM25, hybrid (RRF) and hybrid + cross-encoder rerank."""
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embedder import embed_queries
from .utils import load_chunks, rrf_fuse, tokenize

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class Retriever:
    def __init__(self, chunking: str = "semantic"):
        self.chunking = chunking
        self.chunks = load_chunks(chunking)
        index_path = config.INDEX_DIR / f"{chunking}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found. Run: python -m src.index")
        self.index = faiss.read_index(str(index_path))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None

    # -- individual retrievers: each returns [(chunk_idx, score)] best first -- #
    def dense(self, query: str, n: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(embed_queries([query]), n)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top]

    def hybrid(self, query: str, n: int, pool: int = 50) -> list[tuple[int, float]]:
        rankings = [
            [i for i, _ in self.dense(query, pool)],
            [i for i, _ in self.sparse(query, pool)],
        ]
        return rrf_fuse(rankings)[:n]

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
        return self._reranker

    def rerank(self, query: str, candidates: list[tuple[int, float]], k: int):
        pairs = [(query, self.chunks[i]["text"]) for i, _ in candidates]
        scores = self.reranker.predict(pairs)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(candidates[j][0], float(scores[j])) for j in order]

    # -- public API -- #
    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5,
               rerank_pool: int = 30) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "dense":
            hits = self.dense(query, k)
        elif mode == "bm25":
            hits = self.sparse(query, k)
        elif mode == "hybrid":
            hits = self.hybrid(query, k)
        else:
            hits = self.rerank(query, self.hybrid(query, rerank_pool), k)
        return [{**self.chunks[i], "score": s, "rank": r}
                for r, (i, s) in enumerate(hits, start=1)]
