from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    llm_provider: Literal["openai", "anthropic"] = "openai"
    
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"
    
    embedding_provider: Literal["openai", "local"] = "local"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_chunks: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    
    escalation_webhook_url: str = ""
    escalation_log_file: str = "escalations.jsonl"
    
    host: str = "0.0.0.0"
    port: int = 8742
    debug: bool = False
    
    chroma_persist_dir: str = "./data/chroma"
    knowledge_base_dir: str = "./docs/knowledge_base"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
