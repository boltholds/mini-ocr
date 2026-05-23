from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ocr_db"
    # Local Windows-friendly default. In Docker set STORAGE_DIR=/data/storage.
    storage_dir: Path = Path("data/storage")

    ocr_use_gpu: bool = False
    ocr_lang: str = "ru"
    enable_image_preprocessing: bool = False
    enable_auto_rotation: bool = True

    enable_llm: bool = True
    enable_llm_validation: bool = False
    enable_regex_fallback: bool = False

    # LangGraph orchestrates LLM extraction, RAG validation and status decisions.
    enable_langgraph_workflow: bool = True
    enable_agent_validation: bool = True
    enable_ocr_correction_agent: bool = True
    enable_rag_validation: bool = True
    rag_top_k: int = 5

    # Supported: openai-compatible, ollama
    llm_provider: str = "ollama"
    llm_base_url: str | None = "http://localhost:11434/v1"
    llm_api_key: str | None = "ollama"
    llm_model: str = "gemma3:1b"
    prompt_version: str = "terms_abbrev_extractor_v5_correction_agent"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
