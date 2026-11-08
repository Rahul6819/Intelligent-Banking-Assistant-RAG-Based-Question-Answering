# Bank Policy Q&A — Hybrid RAG with Guardrails and Evaluation

A retrieval-augmented question-answering system over banking policy documents (loans, credit cards,
savings accounts, grievance redressal). Built to show the full GenAI loop: ingestion, retrieval,
grounded generation with citations, guardrails, and **measured** evaluation.

> The sample documents in `data/raw/` are **synthetic** (a fictional "Sample Bank"). Swap in public
> documents (RBI circulars/master directions, bank T&C PDFs) to make it real.

## Architecture

```
question -> PII masking -> injection check -> topic check
         -> BM25 + dense (FAISS) -> Reciprocal Rank Fusion -> cross-encoder rerank
         -> LLM (answer only from numbered passages, must cite [n])
         -> output check (valid citations) -> answer + sources
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | `fixed` (sliding window) vs `section` (heading-aware) | compared in eval |
| Dense retrieval | `BAAI/bge-small-en-v1.5` + FAISS (cosine) | small, strong, CPU-friendly |
| Sparse retrieval | BM25 | exact matches on numbers, fees, product names |
| Fusion | Reciprocal Rank Fusion | no score calibration needed between retrievers |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | precision on the final top-k |
| LLM | `anthropic` / `openai` / `ollama` / `extractive` (no LLM) | `LLM_PROVIDER` env var |

## Run it

```bash
pip install -r requirements.txt
python -m src.eval                      # retrieval ablation (chunking x retrieval mode)
python -m src.eval --guardrails         # guardrail tests
streamlit run app.py                    # UI

# with an LLM:
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...   # or openai / ollama (set LLM_MODEL too)
python -m src.eval --answers            # end-to-end answer check
```

The first run downloads the embedding and reranker models from Hugging Face (needs internet).

## Evaluation

- `data/eval_set.json`: 18 questions, each with a gold fact string. A retrieved chunk counts as
  relevant if it contains that string. Metrics: **hit@4** and **MRR**.
- `data/guardrail_tests.json`: off-topic, prompt injection, unanswerable-but-on-topic, PII.
- `--answers`: answered rate, whether the answer contains the gold fact, and a crude faithfulness
  proxy (numbers in the answer that are absent from the cited passages).

### Results (fill in after your run)

| Chunking | Mode | hit@4 | MRR |
|---|---|---|---|
| fixed | bm25 | | |
| fixed | dense | | |
| fixed | hybrid | | |
| fixed | hybrid_rerank | | |
| section | bm25 | | |
| section | dense | | |
| section | hybrid | | |
| section | hybrid_rerank | | |

Early offline check (BM25 only, no models): section chunking MRR 0.963 vs fixed 0.917, both hit@4 = 18/18.
Use the table above for the full comparison.

## Known limitations (write your own failure analysis here)

- The 18-question eval set is small and the corpus is tiny, so BM25 already scores high. Add more
  documents and harder questions (paraphrases, multi-hop, tables) before trusting the numbers.
- The topic guardrail is a keyword gate; the `MIN_RERANK_SCORE` threshold in `src/pipeline.py`
  must be calibrated on your own data.
- Gold-fact string matching is strict. LLM-as-judge or RAGAS would be a natural next step.
- No PDF table extraction; tables in PDFs will chunk poorly.

## Responsible AI / privacy

PII (card, Aadhaar, PAN, phone, email, IFSC) is masked before retrieval and generation. Answers
must cite sources, and uncited answers are rejected. The model is instructed to refuse when context
is insufficient. No customer data is used.

## Next steps

RAGAS metrics, query rewriting, a FastAPI service with Docker, answer/latency logging, and an A/B
comparison of prompts.
