"""Small pure-python helpers (no heavy dependencies, easy to unit test)."""
import json
import re

from . import config

# NOTE: Python's \w does not match Devanagari vowel signs (matras), which would
# split Hindi words apart. Explicitly include the Devanagari block.
_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_chunks(strategy: str) -> list[dict]:
    path = config.PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.ingest && python -m src.index"
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. `rankings` = ranked id lists (best first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def is_hit(chunk: dict, doc: str, pages: list[int]) -> bool:
    """A retrieved chunk is a hit if it comes from the gold doc and overlaps a gold page.

    Page-level matching keeps the metric independent of the chunking strategy,
    so fixed vs. semantic chunking can be compared fairly.
    """
    if chunk["doc"] != doc:
        return False
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in pages)


def first_hit_rank(chunks: list[dict], doc: str, pages: list[int]) -> int | None:
    """1-based rank of the first hit, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_hit(chunk, doc, pages):
            return rank
    return None
