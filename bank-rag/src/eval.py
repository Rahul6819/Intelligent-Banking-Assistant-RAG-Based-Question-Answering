"""Evaluation harness.

  python -m src.eval                  # retrieval ablation: chunking x retrieval mode
  python -m src.eval --guardrails     # guardrail tests
  python -m src.eval --answers        # end-to-end: citation validity + answer contains gold fact

Retrieval metrics use a gold fact string: a retrieved chunk is relevant if it contains it.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import config
from .pipeline import RAGPipeline
from .retriever import HybridRetriever
from .ingest import build_chunks

DATA = config.ROOT / "data"
RESULTS = config.ROOT / "results"
MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def retrieval_eval(k: int = config.TOP_K) -> pd.DataFrame:
    items = json.loads((DATA / "eval_set.json").read_text())
    rows = []
    for strategy in config.CHUNK_STRATEGIES:
        chunks = build_chunks(strategy)
        retr = HybridRetriever(chunks)
        avg_words = sum(len(c.text.split()) for c in chunks) / len(chunks)
        for mode in MODES:
            hits, rr = 0, 0.0
            for it in items:
                res = retr.search(it["q"], top_k=k, mode=mode)
                rank = next((i for i, (c, _) in enumerate(res, 1) if norm(it["fact"]) in norm(c.text)), None)
                if rank:
                    hits += 1
                    rr += 1 / rank
            rows.append({
                "chunking": strategy, "mode": mode, "n_chunks": len(chunks),
                "avg_words": round(avg_words), f"hit@{k}": round(hits / len(items), 3),
                "MRR": round(rr / len(items), 3),
            })
    return pd.DataFrame(rows)


def guardrail_eval() -> pd.DataFrame:
    pipe = RAGPipeline()
    rows = []
    for t in json.loads((DATA / "guardrail_tests.json").read_text()):
        a = pipe.ask(t["q"])
        ok = a.blocked == t.get("expect_blocked")
        if "expect_pii" in t:
            ok = ok and set(t["expect_pii"]) <= set(a.pii_masked)
        rows.append({"question": t["q"][:70], "expected": t.get("expect_blocked"), "got": a.blocked,
                     "pii_masked": ",".join(a.pii_masked), "pass": ok})
    return pd.DataFrame(rows)


def answer_eval() -> pd.DataFrame:
    pipe = RAGPipeline()
    rows = []
    for it in json.loads((DATA / "eval_set.json").read_text()):
        a = pipe.ask(it["q"])
        cited = a.blocked is None
        # crude faithfulness proxy: all numbers in the answer must appear in cited sources
        nums = set(re.findall(r"\d[\d,.]*", re.sub(r"\[\d+\]", "", a.text)))
        ctx = " ".join(s["text"] for s in a.sources)
        unsupported = [n for n in nums if n.rstrip(".,") not in ctx]
        rows.append({
            "question": it["q"][:60], "answered": cited,
            "contains_fact": norm(it["fact"]).split(" ")[-1] in norm(a.text) or norm(it["fact"]) in norm(a.text),
            "unsupported_numbers": len(unsupported),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", action="store_true")
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.guardrails:
        df = guardrail_eval()
        out = RESULTS / "guardrails.csv"
    elif args.answers:
        df = answer_eval()
        out = RESULTS / f"answers_{config.LLM_PROVIDER}.csv"
        print(f"\nanswered: {df.answered.mean():.0%} | contains fact: {df.contains_fact.mean():.0%} "
              f"| answers with unsupported numbers: {(df.unsupported_numbers > 0).mean():.0%}")
    else:
        df = retrieval_eval()
        out = RESULTS / "retrieval_ablation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
