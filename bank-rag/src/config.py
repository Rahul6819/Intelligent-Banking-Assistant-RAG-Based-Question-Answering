import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking strategies compared in eval (sizes are in words)
CHUNK_STRATEGIES = {
    "fixed": {"size": 120, "overlap": 30},    # sliding word window
    "section": {"max_words": 200},            # split on headings, pack small sections
}
DEFAULT_STRATEGY = "section"

TOP_K = 4              # chunks sent to the LLM
CANDIDATES_K = 20      # candidates pulled before reranking

# LLM: anthropic | openai | ollama | extractive (no LLM, returns best chunk)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
