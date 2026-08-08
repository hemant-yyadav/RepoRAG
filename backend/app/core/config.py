from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings.

    Integration values are declared now for a stable configuration surface, but are
    intentionally unused during Phase 0.
    """

    app_name: str = "Codebase RAG Assistant"
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_model: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = Field(default=None, repr=False)
    max_file_size_bytes: int = 1_048_576
    git_clone_timeout_seconds: int = 60
    max_chunk_size: int = 4_000
    chunk_overlap_lines: int = 2
    min_chunk_size: int = 200
    embedding_provider: str = "openai_compatible"
    embedding_api_key: str | None = Field(default=None, repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 32
    embedding_max_retries: int = 3
    embedding_initial_backoff_seconds: float = 0.5
    qdrant_collection_name: str = "codebase_chunks"
    retrieval_top_k: int = 5
    retrieval_score_threshold: float | None = None
    retrieval_candidate_pool_size: int = 20
    retrieval_rrf_k: int = 60
    retrieval_vector_weight: float = 1.0
    retrieval_bm25_weight: float = 1.0
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidate_count: int = 20
    reranker_final_count: int = 5
    reranker_batch_size: int = 16
    reranker_fail_open: bool = True
    conversation_max_history_messages: int = 12
    conversation_rewrite_history_length: int = 6
    generation_max_context_chars: int = 20_000
    generation_max_context_chunks: int = 12

    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
