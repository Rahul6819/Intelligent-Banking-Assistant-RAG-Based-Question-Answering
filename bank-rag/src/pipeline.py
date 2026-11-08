"""End-to-end pipeline: guardrails -> retrieve -> generate -> output checks."""
from dataclasses import dataclass, field

from . import config, guardrails
from .generator import generate
from .ingest import build_chunks
from .retriever import HybridRetriever

# Cross-encoder logit below which we treat retrieval as "nothing relevant found".
MIN_RERANK_SCORE = -4.0


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    blocked: str | None = None       # why the request was blocked, if it was
    pii_masked: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, strategy: str = config.DEFAULT_STRATEGY):
        self.strategy = strategy
        self.chunks = build_chunks(strategy)
        if not self.chunks:
            raise RuntimeError(f"No documents found in {config.RAW_DIR}")
        self.retriever = HybridRetriever(self.chunks)

    def ask(self, question: str, mode: str = "hybrid_rerank") -> Answer:
        clean_q, pii = guardrails.mask_pii(question)

        if guardrails.is_injection(clean_q):
            return Answer(guardrails.REFUSAL_INJECTION, blocked="prompt_injection", pii_masked=pii)
        if not guardrails.looks_on_topic(clean_q):
            return Answer(guardrails.REFUSAL_OFF_TOPIC, blocked="off_topic", pii_masked=pii)

        hits = self.retriever.search(clean_q, mode=mode)
        if mode == "hybrid_rerank" and (not hits or hits[0][1] < MIN_RERANK_SCORE):
            return Answer(guardrails.NO_ANSWER, blocked="no_relevant_context", pii_masked=pii)

        passages = [c.text for c, _ in hits]
        text = generate(clean_q, passages)

        # Output guardrail: an answer with no valid citation is not trusted.
        if "couldn't find this" not in text and not guardrails.check_citations(text, len(passages)):
            text = guardrails.NO_ANSWER
            return Answer(text, blocked="uncited_answer", pii_masked=pii)

        sources = [
            {"n": i, "doc": c.doc, "section": c.section, "score": round(s, 3), "text": c.text}
            for i, (c, s) in enumerate(hits, 1)
        ]
        return Answer(text, sources, pii_masked=pii)
