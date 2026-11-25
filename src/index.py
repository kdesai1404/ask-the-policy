"""Build FAISS dense indexes from the chunk files. BM25 is built in memory at load time.

Usage:
    python -m src.index                    # both strategies
    python -m src.index --strategies fixed
"""
import argparse

import faiss

from . import config
from .embedder import embed_passages
from .utils import load_chunks


def build(strategy: str) -> None:
    chunks = load_chunks(strategy)
    emb = embed_passages([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])  # exact inner product == cosine (normalised)
    index.add(emb)
    path = config.INDEX_DIR / f"{strategy}.faiss"
    faiss.write_index(index, str(path))
    print(f"[{strategy}] indexed {len(chunks)} chunks (dim={emb.shape[1]}) -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["fixed", "semantic"])
    for s in ap.parse_args().strategies:
        build(s)
