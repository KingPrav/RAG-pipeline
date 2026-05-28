from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Paths
    chroma_persist_dir: str = "./data/chroma"
    bm25_index_path: str = "./data/bm25_index.pkl"
    collection_name: str = "rag_docs"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5

    # Generation
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # Observability
    enable_tracing: bool = False


settings = Settings()
