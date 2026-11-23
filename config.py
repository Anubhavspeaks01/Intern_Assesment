"""
Centralized configuration management for the Healthcare AI Prototype.
Loads environment variables and sets defaults for paths, ML parameters, and RAG settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Set Hugging Face offline environment variables to prevent network latency/timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Settings(BaseSettings):
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    MODELS_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = BASE_DIR / "models" / "best_model.joblib"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    METRICS_PATH: Path = BASE_DIR / "reports" / "metrics.json"

    # ML Hyperparameters & Data Preprocessing
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    IMPUTE_STRATEGY: str = "median"
    ZERO_AS_NAN_COLUMNS: list[str] = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # RAG Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 450
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.35  # Cosine distance or normalized threshold

    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # "gemini", "openai", or "mock"
    GEMINI_API_KEY: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance
settings = Settings()

# Ensure required directories exist
for path in [
    settings.DATA_DIR,
    settings.DATA_DIR / "raw",
    settings.DOCS_DIR,
    settings.MODELS_DIR,
    settings.VECTOR_STORE_DIR,
    settings.REPORTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
