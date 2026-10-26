"""
Core Settings and Configuration for ATLAS Backend.
Loads environment variables and sets defaults for trust thresholds,
Gemini API models, and vector storage parameters.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict
import os
from pathlib import Path

# Base directory for backend
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API & Environment
    PROJECT_NAME: str = "ATLAS: Adaptive Trust Management for Long-term Agent Storage"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # Gemini API Credentials (kept strictly server-side)
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-3.7-flash", env="GEMINI_MODEL")
    GEMINI_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001", env="GEMINI_EMBEDDING_MODEL")
    
    # CORS
    FRONTEND_ORIGIN: str = Field(default="http://localhost:3000", env="FRONTEND_ORIGIN")
    
    # ATLAS Trust Engine Configuration
    ATLAS_TRUST_THRESHOLD: float = Field(default=0.65, env="ATLAS_TRUST_THRESHOLD")
    
    # Trust Signal Weights (Default equal distribution 1/6 each)
    SIGNAL_WEIGHTS: Dict[str, float] = {
        "semantic_relevance": 1.0 / 6.0,
        "embedding_anomaly": 1.0 / 6.0,
        "historical_consistency": 1.0 / 6.0,
        "behavioral_consistency": 1.0 / 6.0,
        "source_reliability": 1.0 / 6.0,
        "temporal_trust": 1.0 / 6.0,
    }
    
    # Source Reliability Defaults (Configurable mapping)
    SOURCE_RELIABILITY_WEIGHTS: Dict[str, float] = {
        "SYSTEM_PROMPT": 1.0,
        "USER_CONVERSATION": 0.90,
        "AGENT_INFERENCE": 0.80,
        "ATTACK_SIMULATOR": 0.25,
    }
    
    # Anomaly Detection Parameters
    ISOLATION_FOREST_CONTAMINATION: float = 0.10
    EMBEDDING_DIMENSION: int = 128
    EMBEDDING_BACKEND: str = Field(default="auto", env="EMBEDDING_BACKEND")   # auto, hash, or local
    MOCK_MODE_FORCED: bool = False

    # Vector storage: memory (default), sqlite, or faiss (needs faiss-cpu)
    VECTOR_STORE_BACKEND: str = Field(default="memory", env="VECTOR_STORE_BACKEND")
    VECTOR_STORE_PATH: str = Field(default="data/atlas_memory.db", env="VECTOR_STORE_PATH")

    # Learned and semantic components
    BEHAV_MODE: str = Field(default="hybrid", env="BEHAV_MODE")             # regex, learned, or hybrid
    HIST_MODE: str = Field(default="clusters", env="HIST_MODE")             # clusters, nli, or hybrid
    ANOMALY_DETECTOR: str = Field(default="iforest", env="ANOMALY_DETECTOR")  # iforest or knn
    POISON_MODEL_PATH: str = Field(default="models/poison_clf", env="POISON_MODEL_PATH")

    # Optional NLI cross-encoder for contradiction checks (needs sentence-transformers)
    USE_NLI: bool = Field(default=False, env="USE_NLI")
    NLI_MODEL: str = Field(default="cross-encoder/nli-deberta-v3-xsmall", env="NLI_MODEL")
    NLI_GATE: str = Field(default="slot", env="NLI_GATE")            # slot, related, or none
    NLI_THRESHOLD: float = Field(default=0.90, env="NLI_THRESHOLD")
    NLI_MIN_SIMILARITY: float = Field(default=0.62, env="NLI_MIN_SIMILARITY")   # semantic NLI modes only
    
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
