from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration with env overrides.

    Environment variables are prefixed with `ZAMEEN_` and a `.env` file is supported.
    """

    start_url: str = Field(
        default="https://www.graana.com/sale/residential-properties-sale-bahria-town-phase-8-rawalpindi-3-258/?pageSize=30&page=1",
        description="Default Graana start URL for scraping",
    )

    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chroma_persist_dir: Path = Path("chromadb_data")

    # Models / RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "rihyesh_listings"

    # HTTP
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    requests_timeout: int = 20

    # LangChain
    langchain_verbose: bool = False
    
    # Gemini API
    gemini_api_key: str = Field(default="", description="Google Gemini API key for LLM inference")

    class Config:
        env_prefix = "GRAANA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables (like Flask-specific ones)


settings = Settings()


