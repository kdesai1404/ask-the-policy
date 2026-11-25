# Evaluation set

Put your questions in `eval/questions.jsonl`, one JSON object per line.

**Answerable question**
```json
{"id": "q001", "question": "Who is eligible for ...?", "answer": "Short reference answer.",
 "doc": "pmkisan_guidelines_en.pdf", "pages": [7, 8], "language": "en", "answerable": true}
```

**Unanswerable question** (the system should refuse)
```json
{"id": "q090", "question": "What is the capital of France?", "answerable": false}
```
Make these *plausible but absent* from your documents (e.g. ask about a scheme you did not include),
not obviously off-topic. They test whether the strict prompt prevents hallucination.

## Suggested composition (~80–100 questions)
- 60–70 answerable, mixed: simple lookups, numbers/dates/thresholds, multi-sentence answers
- 15–20 Hindi questions
- 10–15 unanswerable
- A few that need a table or span two pages

## Workflow
1. `python -m src.gen_eval_candidates --n 100` drafts questions to `eval/candidates.jsonl`
2. Open each one next to the PDF. Fix or delete the question, answer and page numbers.
   Then set `"verified": true`. Merge the keepers into `eval/questions.jsonl`.
3. Add the unanswerable and a few cross-lingual questions by hand.

`pages` are the PDF page numbers (1-based) that contain the answer. Retrieval is scored at the page
level, so fixed and semantic chunking can be compared fairly.
