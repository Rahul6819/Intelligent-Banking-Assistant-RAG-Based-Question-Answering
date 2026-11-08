"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from src import config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Bank Policy Q&A", page_icon="🏦")
st.title("🏦 Bank Policy Q&A (RAG)")
st.caption("Demo over synthetic policy documents. Answers are grounded in retrieved passages with citations.")


@st.cache_resource
def load(strategy: str):
    return RAGPipeline(strategy)


with st.sidebar:
    strategy = st.selectbox("Chunking", list(config.CHUNK_STRATEGIES), index=list(config.CHUNK_STRATEGIES).index(config.DEFAULT_STRATEGY))
    mode = st.selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    st.write(f"LLM provider: `{config.LLM_PROVIDER}`")

pipe = load(strategy)
q = st.text_input("Ask about loans, credit cards, savings accounts or grievances")

if q:
    ans = pipe.ask(q, mode=mode)
    if ans.pii_masked:
        st.warning(f"Sensitive data detected and masked: {', '.join(ans.pii_masked)}")
    if ans.blocked:
        st.info(f"{ans.text}  \n\n_guardrail: {ans.blocked}_")
    else:
        st.markdown(ans.text)
    if ans.sources:
        st.subheader("Sources")
        for s in ans.sources:
            with st.expander(f"[{s['n']}] {s['doc']} — {s['section']} (score {s['score']})"):
                st.write(s["text"])
