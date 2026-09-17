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
    MOCK_MODE_FORCED: bool = False
    
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
