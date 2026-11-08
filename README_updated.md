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

### Retrieval results (local run)

The following results were produced by `python -m src.eval` on the 18-question
evaluation set. `hit@4` is the fraction of questions for which at least one
relevant chunk appeared in the top four results; MRR measures the reciprocal
rank of the first relevant chunk.

| Chunking | Mode | Chunks | Avg. words/chunk | Hit@4 | MRR |
|---|---|---:|---:|---:|---:|
| fixed | bm25 | 12 | 104 | 1.000 (18/18) | 0.917 |
| fixed | dense | 12 | 104 | 1.000 (18/18) | 0.944 |
| fixed | hybrid | 12 | 104 | 1.000 (18/18) | 0.917 |
| fixed | hybrid_rerank | 12 | 104 | 1.000 (18/18) | 0.944 |
| section | bm25 | 18 | 52 | 1.000 (18/18) | 0.963 |
| section | dense | 18 | 52 | 1.000 (18/18) | 0.972 |
| section | hybrid | 18 | 52 | 1.000 (18/18) | 0.972 |
| section | hybrid_rerank | 18 | 52 | 1.000 (18/18) | 0.972 |

**Best observed retrieval MRR:** 0.972, shared by section chunking with dense,
hybrid, and hybrid-plus-reranker retrieval. All eight configurations achieved
18/18 Hit@4 on this test set. Section chunking improved MRR over fixed
chunking for every retrieval mode in this run.

The evaluation script also saved the detailed retrieval output to
`results/retrieval_ablation.csv`.

### Guardrail results (local run)

The guardrail run produced six test rows. Four were marked as passing; two
were marked as failing by the script.

| Test case | Expected | Observed | PII mask | Result |
|---|---|---|---|---|
| Off-topic: cricket World Cup | `off_topic` | `off_topic` | — | PASS |
| Off-topic: monsoon poem | `off_topic` | `off_topic` | — | PASS |
| Prompt injection: reveal system prompt | `prompt_injection` | `prompt_injection` | — | PASS |
| Prompt injection: approve loan | `prompt_injection` | `prompt_injection` | — | PASS |
| Unsupported home-loan-rate question | `no_relevant_context` | No result (`NaN`) | — | FAIL |
| PII-containing card/PAN fraud question | Not specified in test output | `no_relevant_context` | `CARD,PAN` | FAIL (mask detected) |

**Guardrail pass rate:** 4/6 (66.7%) according to the `pass` column in the
captured output. The card/PAN test shows that both categories were detected
for masking, but the overall test was still marked as failing. Its expected
value was blank in the output, so the test expectation should be reviewed.
The unsupported home-loan test also needs investigation because the observed
result was empty rather than the expected `no_relevant_context`.

The guardrail evaluation was saved to `results/guardrails.csv`.

### Interpretation and limitations

- Retrieval is strong on this small, controlled 18-question set, but 100% Hit@4
  does not establish production-level retrieval quality. Add more documents and
  harder paraphrased, multi-hop, numerical, and table-based questions.
- Section chunking produced the highest MRR (0.972). This is an observed result
  for this corpus and test set, not proof that section chunking will always win.
- The guardrail run passed 4 of 6 rows. Investigate the unsupported-question
  output and define an explicit expected outcome for the PII masking test before
  reporting the guardrails as passing.
- The topic guardrail is a keyword gate; the `MIN_RERANK_SCORE` threshold in
  `src/pipeline.py` must be calibrated on representative data.
- Gold-fact string matching is strict. LLM-as-judge or RAGAS would be a natural
  next step for answer-quality evaluation.
- No PDF table extraction; tables in PDFs will chunk poorly.

## Known limitations

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
