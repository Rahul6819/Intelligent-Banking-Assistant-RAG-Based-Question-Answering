"""Hybrid retrieval: BM25 + dense (FAISS), fused with Reciprocal Rank Fusion, then cross-encoder rerank."""
import re
from functools import lru_cache

import numpy as np

from . import config
from .ingest import Chunk


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        import faiss
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([_tok(c.text) for c in chunks])
        emb = _embedder().encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized inner product
        self.index.add(emb)

    # --- individual retrievers (kept separate so eval can ablate them) ---
    def bm25_search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(_tok(query))
        return list(np.argsort(-scores)[:k])

    def dense_search(self, query: str, k: int) -> list[int]:
        q = _embedder().encode(
            [config.BGE_QUERY_PREFIX + query], normalize_embeddings=True
        ).astype("float32")
        _, ids = self.index.search(q, min(k, len(self.chunks)))
        return list(ids[0])

    @staticmethod
    def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
        score: dict[int, float] = {}
        for ranks in rank_lists:
            for r, idx in enumerate(ranks):
                score[idx] = score.get(idx, 0.0) + 1.0 / (k + r + 1)
        return sorted(score, key=score.get, reverse=True)

    def search(self, query: str, top_k: int = config.TOP_K, mode: str = "hybrid_rerank"):
        """mode: bm25 | dense | hybrid | hybrid_rerank. Returns list[(Chunk, score)]."""
        n = config.CANDIDATES_K
        if mode == "bm25":
            ids = self.bm25_search(query, top_k)
        elif mode == "dense":
            ids = self.dense_search(query, top_k)
        else:
            fused = self.rrf([self.bm25_search(query, n), self.dense_search(query, n)])[:n]
            if mode == "hybrid":
                ids = fused[:top_k]
            else:
                pairs = [(query, self.chunks[i].text) for i in fused]
                scores = _reranker().predict(pairs)
                order = np.argsort(-scores)[:top_k]
                return [(self.chunks[fused[o]], float(scores[o])) for o in order]
        return [(self.chunks[i], 0.0) for i in ids]
