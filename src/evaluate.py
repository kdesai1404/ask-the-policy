"""Ablation study: retrieval metrics + LLM-judged faithfulness / correctness.

Usage:
    python -m src.evaluate --eval eval/questions.jsonl --retrieval-only   # fast, no LLM calls
    python -m src.evaluate --eval eval/questions.jsonl                    # full run
    python -m src.evaluate --eval eval/questions.jsonl --configs 0 3 4

Eval file format (JSONL), one question per line:
    {"id": "q001", "question": "...", "answer": "reference answer", "doc": "file.pdf",
     "pages": [12, 13], "language": "en", "answerable": true}
    {"id": "q090", "question": "...", "answerable": false}       # should be refused
Entries with "verified": false are skipped unless --include-unverified is passed.
"""
import argparse
import json
import re
import statistics

from tqdm import tqdm

from . import config, generate, llm
from .retrieve import Retriever
from .utils import first_hit_rank

CONFIGS = [
    {"name": "Dense only",                                  "chunking": "fixed",    "mode": "dense",         "strict": False},
    {"name": "BM25 only",                                   "chunking": "fixed",    "mode": "bm25",          "strict": False},
    {"name": "Hybrid (RRF)",                                "chunking": "fixed",    "mode": "hybrid",        "strict": False},
    {"name": "Hybrid + rerank",                             "chunking": "fixed",    "mode": "hybrid_rerank", "strict": False},
    {"name": "Hybrid + rerank + strict prompt",             "chunking": "fixed",    "mode": "hybrid_rerank", "strict": True},
    {"name": "Semantic chunks + hybrid + rerank + strict",  "chunking": "semantic", "mode": "hybrid_rerank", "strict": True},
]

K_GEN = 5

FAITH_PROMPT = """You are grading whether an answer is supported by the given context.
Context passages:
{context}

Answer to grade:
{answer}

Score the answer's faithfulness to the context:
1   = every factual claim in the answer is supported by the context
0.5 = some claims are supported, but at least one is not supported by the context
0   = most claims are unsupported or contradict the context
Return JSON only: {{"score": <0|0.5|1>, "reason": "<one short sentence>"}}"""

CORRECT_PROMPT = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Score the candidate's correctness:
1   = conveys the same key facts as the reference
0.5 = partially correct or incomplete
0   = incorrect, irrelevant, or says the information is not available
Return JSON only: {{"score": <0|0.5|1>, "reason": "<one short sentence>"}}"""


def parse_score(text: str):
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        return float(json.loads(m.group())["score"])
    except Exception:
        return None


def judge_faithfulness(answer: str, chunks: list[dict]):
    prompt = FAITH_PROMPT.format(context=generate.format_context(chunks), answer=answer)
    return parse_score(llm.chat([{"role": "user", "content": prompt}], max_tokens=150))


def judge_correctness(question: str, reference: str, answer: str):
    prompt = CORRECT_PROMPT.format(question=question, reference=reference, answer=answer)
    return parse_score(llm.chat([{"role": "user", "content": prompt}], max_tokens=150))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load_questions(path: str, include_unverified: bool) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    if not include_unverified:
        qs = [q for q in qs if q.get("verified", True)]
    return qs


def run_config(cfg: dict, questions: list[dict], retriever: Retriever, generation: bool):
    rows = []
    for q in tqdm(questions, desc=cfg["name"], leave=False):
        answerable = q.get("answerable", True)
        results = retriever.search(q["question"], mode=cfg["mode"], k=10)
        row = {"id": q["id"], "question": q["question"], "answerable": answerable,
               "language": q.get("language", "?")}
        if answerable:
            row["rank"] = first_hit_rank(results, q["doc"], q["pages"])
        if generation:
            gen = generate.answer(q["question"], results[:K_GEN], strict=cfg["strict"])
            row.update(answer=gen["raw_answer"], refused=gen["refused"])
            if not gen["refused"]:
                row["faithfulness"] = judge_faithfulness(gen["raw_answer"], results[:K_GEN])
            if answerable:
                row["correctness"] = 0.0 if gen["refused"] else judge_correctness(
                    q["question"], q["answer"], gen["raw_answer"])
        rows.append(row)
    return rows


def summarize(rows: list[dict], generation: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    out = {"n_answerable": len(ans), "n_unanswerable": len(unans)}
    for k in (1, 3, 5):
        out[f"hit@{k}"] = mean([1.0 if r["rank"] and r["rank"] <= k else 0.0 for r in ans])
    out["mrr@10"] = mean([1.0 / r["rank"] if r["rank"] else 0.0 for r in ans])
    if generation:
        out["faithfulness"] = mean([r.get("faithfulness") for r in rows if not r.get("refused")])
        out["correctness"] = mean([r.get("correctness") for r in ans])
        out["false_refusal_rate"] = mean([1.0 if r["refused"] else 0.0 for r in ans])
        out["correct_refusal_rate"] = mean([1.0 if r["refused"] else 0.0 for r in unans])
    return out


def fmt(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def to_markdown(results: list[dict], generation: bool) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "mrr@10"]
    if generation:
        cols += ["faithfulness", "correctness", "false_refusal_rate", "correct_refusal_rate"]
    header = "| Config | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [header, sep]
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['config']['name']} | " + " | ".join(fmt(s.get(c)) for c in cols) + " |")
    n = results[0]["summary"]
    lines.append(f"\n_{n['n_answerable']} answerable + {n['n_unanswerable']} unanswerable questions; "
                 f"judge/generator model: {config.LLM_MODEL}_")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(config.ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--configs", nargs="+", type=int, help="indices into CONFIGS")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--include-unverified", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.eval, args.include_unverified)
    if args.limit:
        questions = questions[: args.limit]
    generation = not args.retrieval_only
    selected = [CONFIGS[i] for i in args.configs] if args.configs else CONFIGS
    print(f"{len(questions)} questions, {len(selected)} configs, generation={generation}")

    retrievers: dict[str, Retriever] = {}
    results = []
    for cfg in selected:
        retriever = retrievers.setdefault(cfg["chunking"], Retriever(cfg["chunking"]))
        rows = run_config(cfg, questions, retriever, generation)
        results.append({"config": cfg, "summary": summarize(rows, generation), "rows": rows})
        print(f"  done: {cfg['name']}")

    suffix = "retrieval" if args.retrieval_only else "full"
    (config.RESULTS_DIR / f"results_{suffix}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(results, generation)
    (config.RESULTS_DIR / f"results_{suffix}.md").write_text(md, encoding="utf-8")

    # Failure analysis: retrieval misses and wrong answers, per config
    with open(config.RESULTS_DIR / f"failures_{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            for row in r["rows"]:
                missed = row["answerable"] and not (row.get("rank") and row["rank"] <= 5)
                wrong = generation and row["answerable"] and (row.get("correctness") or 0) < 1
                bad_refusal = generation and not row["answerable"] and not row.get("refused")
                if missed or wrong or bad_refusal:
                    f.write(json.dumps({"config": r["config"]["name"], **row},
                                       ensure_ascii=False) + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
