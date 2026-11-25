"""Screen PDFs for garbled text (legacy-font Hindi) BEFORE ingesting them.

Usage (from the project folder):
    python check_pdfs.py data\\raw            # every PDF in the folder
    python check_pdfs.py some_file.pdf       # one file

Verdicts:
    OK         readable text
    GARBLED    Devanagari mixed with junk Latin symbols -> do not use this file
    NO TEXT    scanned / image-only -> needs OCR, skip it
"""
import re
import sys
from pathlib import Path

DEVA = re.compile(r"[ऀ-ॿ]")
# Junk produced by broken font maps: accented/extended Latin letters and symbols
# (U+00A0-U+024F) that never appear in normal English or Hindi policy text.
JUNK = re.compile(r"[ -ɏ -⁯]")
PUNCT_OK = set(" –—‘’“”•…")  # nbsp, dashes, quotes, bullet


def score(text: str) -> dict:
    deva = len(DEVA.findall(text))
    junk = sum(1 for c in JUNK.findall(text) if c not in PUNCT_OK)
    letters = len(re.findall(r"[A-Za-z]", text))
    if deva >= 200:  # Hindi document: compare junk to Devanagari
        ratio = junk / max(deva, 1)
    else:  # English document: junk relative to letters
        ratio = junk / max(letters + deva, 1)
    if deva + letters < 200:
        verdict = "NO TEXT"
    elif ratio > 0.03:
        verdict = "GARBLED"
    else:
        verdict = "OK"
    return {"devanagari": deva, "latin": letters, "junk": junk, "ratio": ratio, "verdict": verdict}


def check(pdf: Path) -> dict:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    return score(text)


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not pdfs:
        sys.exit(f"No PDFs found in {target}")
    print(f"{'File':55} {'Devanagari':>10} {'Junk':>7} {'Junk%':>7}  Verdict")
    for p in pdfs:
        r = check(p)
        print(f"{p.name[:55]:55} {r['devanagari']:>10} {r['junk']:>7} {r['ratio'] * 100:>6.1f}%  {r['verdict']}")


if __name__ == "__main__":
    main()
