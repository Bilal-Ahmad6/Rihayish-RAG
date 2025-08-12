from pathlib import Path
from pydantic import Field

# Graceful import of BaseSettings so the app does not crash if pydantic_settings
# was accidentally omitted from the environment. We still strongly recommend
# having `pydantic-settings` installed (added back to requirements.txt). If it's
# missing, we fall back to a lightweight shim that reads a .env file manually.
try:  # pragma: no cover - runtime safeguard
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:  # pragma: no cover
    import os
    import warnings
    warnings.warn("pydantic_settings not installed; using fallback BaseSettings shim."
                  " Install pydantic-settings for full features.")

    class BaseSettings:  # minimal shim
        def __init__(self, **kwargs):
            # Load .env manually (very simple parser)
            env_file = kwargs.pop('env_file', '.env') if 'env_file' in kwargs else '.env'
            data = {}
            if Path(env_file).exists():
                for line in Path(env_file).read_text().splitlines():
                    if not line or line.strip().startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
            # Merge environment variables
            data.update(os.environ)
            # Assign provided overrides
            data.update(kwargs)
            for field, value in data.items():
                setattr(self, field.lower(), value)


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


