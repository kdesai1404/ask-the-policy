"""Draft candidate eval questions from random chunks to speed up building the eval set.

IMPORTANT: these are *drafts*. Review every one against the PDF, fix the question/answer/pages,
then set "verified": true. Only verified questions are used by src.evaluate. Also write
10-15 unanswerable questions by hand (set "answerable": false, no doc/pages).

Usage:
    python -m src.gen_eval_candidates --n 100 --out eval/candidates.jsonl
"""
import argparse
import json
import random
import re

from . import config, llm
from .utils import load_chunks

PROMPT = """Below is a passage from a government policy document.
Write ONE specific question that a citizen might ask, which is answered by this passage alone,
and a short factual answer. Write both in {lang}. Avoid vague questions like "what is this document about".
Return JSON only: {{"question": "...", "answer": "..."}}

Passage:
{text}"""


def is_hindi(text: str) -> bool:
    deva = len(re.findall(r"[ऀ-ॿ]", text))
    return deva / max(len(text), 1) > 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(config.ROOT / "eval" / "candidates.jsonl"))
    args = ap.parse_args()

    random.seed(args.seed)
    pool = [c for c in load_chunks("fixed") if len(c["text"]) >= args.min_chars]
    random.shuffle(pool)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for c in pool:
            if written >= args.n:
                break
            lang = "Hindi" if is_hindi(c["text"]) else "English"
            raw = llm.chat([{"role": "user", "content": PROMPT.format(lang=lang, text=c["text"])}])
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                qa = json.loads(m.group())
                question, answer = qa["question"].strip(), qa["answer"].strip()
            except Exception:
                continue
            written += 1
            f.write(json.dumps({
                "id": f"q{written:03d}",
                "question": question,
                "answer": answer,
                "doc": c["doc"],
                "pages": list(range(c["page_start"], c["page_end"] + 1)),
                "evidence": c["text"],
                "language": "hi" if lang == "Hindi" else "en",
                "answerable": True,
                "verified": False,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {written} draft questions to {args.out}. Review them, then set verified=true.")


if __name__ == "__main__":
    main()
