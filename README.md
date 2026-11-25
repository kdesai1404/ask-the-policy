# Ask-the-Policy

Multilingual (English + Hindi) question answering over Indian government policy documents, with a
**measured evaluation** of every design choice: chunking, dense vs. BM25 vs. hybrid retrieval,
cross-encoder reranking, and a strict "refuse if unsure" prompt.

> Status: pipeline complete; results below are filled in after running the evaluation (see "Reproduce").

## Architecture

```
PDFs ─► parse + clean ─► chunk (fixed | semantic) ─► embeddings (multilingual-e5) ─► FAISS
                                                  └► BM25
query ─► dense + BM25 ─► RRF fusion ─► cross-encoder rerank ─► top-k ─► LLM (cited, may refuse) ─► answer
```

| Layer | Implementation |
|---|---|
| Parsing | PyMuPDF; header/footer + page-number stripping; scanned-page detection |
| Chunking | Fixed-size with overlap vs. semantic (split on sentence-similarity drops, Hindi danda aware) |
| Retrieval | `multilingual-e5-small` + FAISS, BM25 (Devanagari-safe tokenizer), Reciprocal Rank Fusion |
| Reranking | Multilingual cross-encoder (`mmarco-mMiniLMv2`), configurable |
| Generation | Any OpenAI-compatible LLM; numbered citations; `NOT_FOUND` abstention |
| Evaluation | Hit@k, MRR (page-level gold), LLM-judged faithfulness and correctness, refusal rates |
| Serving | FastAPI (`/ask`), Streamlit UI, Docker + docker-compose, CI running unit tests |

## Results

Eval set: **N** questions (**A** answerable, **U** unanswerable, **H** Hindi) over **D** documents / **P** pages.

| Config | Hit@1 | Hit@5 | MRR@10 | Faithfulness | Correctness | Correct refusals |
|---|---|---|---|---|---|---|
| Dense only | – | – | – | – | – | – |
| BM25 only | – | – | – | – | – | – |
| Hybrid (RRF) | – | – | – | – | – | – |
| Hybrid + rerank | – | – | – | – | – | – |
| Hybrid + rerank + strict prompt | – | – | – | – | – | – |
| Semantic chunks + hybrid + rerank + strict | – | – | – | – | – | – |

Paste the table from `results/results_full.md` here.

### Failure analysis
_Write 4–6 bullets after reading `results/failures_full.jsonl`. For example: which question types fail,
whether Hindi lags English, table-heavy pages, OCR/legacy-font issues, over-refusal. Be specific._

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your LLM_API_KEY

# 1. add PDFs to data/raw/ (see data/raw/README.md)
python -m src.ingest            # parse + chunk (fixed and semantic)
python -m src.index             # build FAISS indexes

# 2. try it
uvicorn src.api:app --reload    # API  -> http://localhost:8000/docs
streamlit run app.py            # UI   -> http://localhost:8501
```

## Reproduce the evaluation

```bash
# build eval/questions.jsonl (see eval/README.md), then:
python -m src.evaluate --retrieval-only   # fast retrieval metrics, no LLM calls
python -m src.evaluate                    # full ablation incl. faithfulness/correctness
```
Outputs land in `results/`. LLM calls are cached in `data/cache/`, so re-runs are cheap.

## Docker

```bash
docker compose up --build     # API :8000, UI :8501
```

## Limitations
- LLM-as-judge scores are noisy and depend on the judge model; spot-check a sample by hand.
- Scanned PDFs need OCR (not included). Legacy-font Hindi PDFs may extract as garbage.
- Tables are flattened to text, which hurts questions about tabular data.

## Document sources
_List each PDF with its source URL and download date._
