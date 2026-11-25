"""PDF -> cleaned pages -> chunks (two strategies: fixed | semantic).

Usage:
    python -m src.ingest                       # both strategies
    python -m src.ingest --strategies fixed
"""
import argparse
import bisect
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import config

# Sentence splitter that also handles the Hindi danda (।)
SENT_RE = re.compile(r"[^.!?।]+[.!?।]*\s*")


# --------------------------------------------------------------------------- #
# Parsing and cleaning
# --------------------------------------------------------------------------- #
def clean_text(t: str) -> str:
    t = t.replace("­", "")  # soft hyphens
    t = re.sub(r"-\n(?=\w)", "", t)  # de-hyphenate words broken across lines
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)  # single newline -> space
    t = re.sub(r"\n{2,}", "\n\n", t)  # keep paragraph breaks
    t = re.sub(r"[ \t ]+", " ", t)
    return t.strip()


def strip_boilerplate(raw_pages: list[str]) -> list[str]:
    """Drop running headers/footers (short lines repeated on >=30% of pages) and page numbers."""
    n = len(raw_pages)
    common: set[str] = set()
    if n >= 5:
        counts = Counter(
            line
            for text in raw_pages
            for line in {x.strip() for x in text.splitlines() if 0 < len(x.strip()) < 100}
        )
        common = {line for line, c in counts.items() if c / n >= 0.3}
    page_no = re.compile(r"\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*", re.I)
    cleaned = []
    for text in raw_pages:
        keep = [
            ln for ln in text.splitlines()
            if ln.strip() not in common and not page_no.fullmatch(ln)
        ]
        cleaned.append("\n".join(keep))
    return cleaned


def extract_pages(pdf_path: Path) -> list[dict]:
    import pymupdf  # PyMuPDF

    with pymupdf.open(pdf_path) as pdf:
        raw = [page.get_text("text") for page in pdf]
    pages, skipped = [], 0
    for i, text in enumerate(strip_boilerplate(raw), start=1):
        text = clean_text(text)
        if len(text) < 20:
            skipped += 1  # likely a scanned page (needs OCR) or an image-only page
            continue
        pages.append({"doc": pdf_path.name, "page": i, "text": text})
    if skipped:
        print(f"  [warn] {pdf_path.name}: {skipped} page(s) had no extractable text (scanned? needs OCR)")
    return pages


def build_stream(pages: list[dict]):
    """Join pages into one text stream, remembering where each page starts."""
    parts, spans, pos = [], [], 0
    for p in pages:
        spans.append((pos, p["page"]))
        chunk = p["text"] + "\n\n"
        parts.append(chunk)
        pos += len(chunk)
    return "".join(parts), spans


def page_lookup(spans):
    offsets = [o for o, _ in spans]
    pages = [p for _, p in spans]

    def at(pos: int) -> int:
        return pages[max(bisect.bisect_right(offsets, pos) - 1, 0)]

    return at


# --------------------------------------------------------------------------- #
# Chunking strategies (both return (start, end) character spans)
# --------------------------------------------------------------------------- #
def fixed_spans(text: str, size: int = 900, overlap: int = 150) -> list[tuple[int, int]]:
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # snap the end back to a word boundary
            cut = text.rfind(" ", start + int(size * 0.6), end)
            if cut != -1:
                end = cut
        out.append((start, end))
        if end >= n:
            break
        start = end - overlap
        nxt = text.find(" ", start, end)  # snap the start forward to a word boundary
        if nxt != -1:
            start = nxt + 1
    return out


def sentence_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in SENT_RE.finditer(text) if m.group().strip()]


def semantic_spans(
    text: str, min_chars: int = 300, max_chars: int = 1200, percentile: float = 20
) -> list[tuple[int, int]]:
    """Start a new chunk where adjacent sentences are least similar (topic shift)."""
    from .embedder import embed_passages  # lazy: heavy import

    sents = sentence_spans(text)
    if len(sents) < 3:
        return [(0, len(text))] if text.strip() else []
    emb = embed_passages([text[s:e].strip() for s, e in sents])
    sims = (emb[:-1] * emb[1:]).sum(axis=1)  # cosine (embeddings are normalised)
    threshold = float(np.percentile(sims, percentile))

    spans = []
    cur_start, cur_end = sents[0]
    for i in range(1, len(sents)):
        s, e = sents[i]
        topic_shift = sims[i - 1] < threshold and (cur_end - cur_start) >= min_chars
        too_long = (e - cur_start) > max_chars
        if topic_shift or too_long:
            spans.append((cur_start, cur_end))
            cur_start = s
        cur_end = e
    spans.append((cur_start, cur_end))
    return spans


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--strategies", nargs="+", default=["fixed", "semantic"],
                    choices=["fixed", "semantic"])
    ap.add_argument("--size", type=int, default=900, help="fixed: chunk size in characters")
    ap.add_argument("--overlap", type=int, default=150, help="fixed: overlap in characters")
    ap.add_argument("--min-chars", type=int, default=300, help="semantic: min chunk size")
    ap.add_argument("--max-chars", type=int, default=1200, help="semantic: max chunk size")
    ap.add_argument("--percentile", type=float, default=20,
                    help="semantic: split where similarity is below this percentile")
    args = ap.parse_args()

    pdfs = sorted(Path(args.raw).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.raw}. Add documents first (see data/raw/README.md).")

    docs = []
    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        pages = extract_pages(pdf)
        if pages:
            docs.append((pdf.name, pages))
    print(f"Parsed {len(docs)} documents, {sum(len(p) for _, p in docs)} pages")

    for strategy in args.strategies:
        chunks = []
        for name, pages in docs:
            text, spans = build_stream(pages)
            at = page_lookup(spans)
            if strategy == "fixed":
                ranges = fixed_spans(text, args.size, args.overlap)
            else:
                ranges = semantic_spans(text, args.min_chars, args.max_chars, args.percentile)
            for j, (s, e) in enumerate(ranges):
                piece = text[s:e].strip()
                if len(piece) < 40:
                    continue
                chunks.append({
                    "chunk_id": f"{name}::{strategy}::{j}",
                    "doc": name,
                    "page_start": at(s),
                    "page_end": at(max(e - 1, s)),
                    "text": piece,
                    "strategy": strategy,
                })
        out = config.PROCESSED_DIR / f"chunks_{strategy}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        avg = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
        print(f"[{strategy}] {len(chunks)} chunks, avg {avg:.0f} chars -> {out}")


if __name__ == "__main__":
    main()
