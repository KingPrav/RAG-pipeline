# RAG Pipeline

Production-style retrieval-augmented generation (RAG) with **hybrid search** (BM25 + dense embeddings), **cross-encoder reranking**, and **Claude** answer generation. Includes a FastAPI backend, Streamlit demo, and optional RAGAS evaluation.

## Features

- **Hybrid retrieval** — combine sparse (BM25) and dense (Chroma + OpenAI embeddings) results
- **Reranking** — cross-encoder reranker narrows candidates before generation
- **Ingestion** — load markdown from a local folder or shallow-clone a GitHub repo
- **API** — `/query` and `/query/stream` (SSE) endpoints
- **UI** — Streamlit app to compare hybrid, dense-only, and sparse-only modes
- **Evaluation** — optional RAGAS metrics (`pip install -e ".[eval]"`)

## Requirements

- Python 3.11+
- [Anthropic](https://console.anthropic.com/) API key (generation)
- [OpenAI](https://platform.openai.com/) API key (embeddings)

## Setup

```bash
cd RAG
pip install -e .
```

Create a `.env` file in the project root (not committed to git):

```env
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

Optional overrides: `CLAUDE_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_DENSE`, `TOP_K_SPARSE`, `TOP_K_RERANK`, `ENABLE_TRACING`.

## Ingest documents

Sample docs are included under `data/sample/`. To build the index:

```bash
python scripts/ingest.py --dir data/sample
```

Ingest from a GitHub repo (use the repo’s default branch name, e.g. `master` for FastAPI):

```bash
python scripts/ingest.py --github https://github.com/tiangolo/fastapi --branch master --subdir docs/en/docs
```

Indexes are written to `data/chroma/` and `data/bm25_index.pkl`.

## Run

**Streamlit demo**

```bash
streamlit run app/streamlit_app.py
```

**API server**

```bash
uvicorn api.main:app --reload --port 8000
```

- Health: `GET http://localhost:8000/health`
- Query: `POST http://localhost:8000/query` with JSON body matching `RAGRequest`
- Stream: `POST http://localhost:8000/query/stream`

**RAGAS evaluation** (optional)

```bash
pip install -e ".[eval]"
python scripts/evaluate.py
python scripts/evaluate.py --mode hybrid   # or dense, sparse
```

## Project layout

```
api/           FastAPI application
app/           Streamlit UI
scripts/       ingest.py, evaluate.py
src/
  ingestion/   loaders, chunking, indexing
  retrieval/   dense, sparse, hybrid, reranker
  generation/  prompts and Claude client
  evaluation/  RAGAS harness
data/          sample docs and generated indexes (local)
```

## License

MIT (or your chosen license).
