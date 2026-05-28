"""
Hybrid RAG Pipeline — Interactive Demo

Compares hybrid (BM25 + dense + rerank), dense-only, and sparse-only retrieval
side by side so you can see concretely why each component adds value.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.pipeline import RAGPipeline
from src.models import RAGRequest, RAGResponse

st.set_page_config(
    page_title="Hybrid RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["hybrid", "dense", "sparse"],
        help=(
            "**hybrid**: BM25 + dense embeddings fused with RRF, then cross-encoder reranked\n\n"
            "**dense**: OpenAI embeddings + cosine similarity only\n\n"
            "**sparse**: BM25 keyword matching only"
        ),
    )
    top_k = st.slider("Final top-K sources", min_value=1, max_value=10, value=5)
    show_debug = st.toggle("Show retrieval pipeline debug", value=True)
    compare_all = st.toggle("Compare all 3 modes side-by-side", value=False)

    st.divider()
    st.caption("**Stack**")
    st.caption("• BM25 (`rank-bm25`)")
    st.caption("• Dense: OpenAI `text-embedding-3-small` + ChromaDB")
    st.caption("• Fusion: Reciprocal Rank Fusion (k=60)")
    st.caption("• Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`")
    st.caption("• LLM: Claude via Anthropic SDK")


# ── Pipeline (cached across reruns) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🔍 Hybrid RAG Pipeline")
st.caption(
    "BM25 + dense retrieval fused with Reciprocal Rank Fusion · "
    "cross-encoder reranking · Claude generation with citations"
)

query = st.text_input(
    "Ask a question:",
    placeholder="e.g. How does asyncio.gather work?",
)

if not query:
    st.info("Enter a question above. Ingest documents first with `python scripts/ingest.py --dir data/sample`.")
    st.stop()

pipeline = load_pipeline()


def render_response(response: RAGResponse, label: str = "") -> None:
    if label:
        st.subheader(label)

    st.markdown(response.answer)

    # Metrics row
    cols = st.columns(4)
    cols[0].metric("Latency", f"{response.latency_ms:.0f} ms")
    cols[1].metric("Sources", len(response.sources))
    if response.retrieval_debug.dense_count:
        cols[2].metric("Dense candidates", response.retrieval_debug.dense_count)
    if response.retrieval_debug.sparse_count:
        cols[3].metric("Sparse candidates", response.retrieval_debug.sparse_count)

    # Retrieval debug
    if show_debug and response.retrieval_debug.mode == "hybrid":
        with st.expander("Retrieval pipeline breakdown"):
            d = response.retrieval_debug
            c1, c2, c3 = st.columns(3)
            c1.metric("BM25 unique hits", d.sparse_only_count, help="Retrieved by BM25 only")
            c2.metric("Dense unique hits", d.dense_only_count, help="Retrieved by dense only")
            c3.metric("Overlap", d.overlap_count, help="Retrieved by both — highest-confidence candidates")
            st.caption(
                f"RRF merged {d.dense_count} dense + {d.sparse_count} sparse "
                f"→ {d.fused_count} unique → cross-encoder reranked → top {len(response.sources)}"
            )

    # Retrieved sources
    st.subheader("Sources")
    for chunk in response.sources:
        source = chunk.document.metadata.get("source", "unknown")
        node_name = chunk.document.metadata.get("node_name", "")
        title = f"[{chunk.rank}] {os.path.basename(source)}"
        if node_name:
            title += f" · `{node_name}`"
        title += f"  (score: {chunk.score:.4f})"

        with st.expander(title):
            ext = chunk.document.metadata.get("extension", "")
            lang = "python" if ext == ".py" else "markdown" if ext in (".md", ".rst") else "text"
            st.code(chunk.document.content, language=lang)
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"Method: `{chunk.method}`")
            meta_cols[1].caption(f"Rank: {chunk.rank}")
            meta_cols[2].caption(f"Source: `{source}`")


if compare_all:
    # Run all three modes and render in columns
    with st.spinner("Running all three retrieval modes..."):
        responses = {
            mode: pipeline.query(RAGRequest(query=query, top_k=top_k, retrieval_mode=mode))
            for mode in ("hybrid", "dense", "sparse")
        }

    tab_hybrid, tab_dense, tab_sparse = st.tabs(["Hybrid (RRF + rerank)", "Dense only", "Sparse (BM25) only"])
    with tab_hybrid:
        render_response(responses["hybrid"])
    with tab_dense:
        render_response(responses["dense"])
    with tab_sparse:
        render_response(responses["sparse"])
else:
    with st.spinner("Retrieving and generating..."):
        response = pipeline.query(RAGRequest(query=query, top_k=top_k, retrieval_mode=retrieval_mode))
    render_response(response)
