| Config | hit@1 | hit@3 | hit@5 | mrr@10 |
|---|---|---|---|---|
| Dense only | 60.8% | 78.4% | 85.1% | 70.8% |
| BM25 only | 58.1% | 81.1% | 86.5% | 70.1% |
| Hybrid (RRF) | 71.6% | 83.8% | 89.2% | 78.8% |
| Hybrid + rerank | 83.8% | 94.6% | 98.6% | 89.5% |
| Hybrid + rerank + strict prompt | 83.8% | 94.6% | 98.6% | 89.5% |
| Semantic chunks + hybrid + rerank + strict | 79.7% | 90.5% | 95.9% | 86.0% |

_74 answerable + 14 unanswerable questions; judge/generator model: openai/gpt-oss-120b_