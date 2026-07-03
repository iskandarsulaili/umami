"""
Umami ML Service Configuration
Central config with GPU-aware defaults
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """PostgreSQL/TimescaleDB connection"""
    host: str = os.getenv("DATABASE_HOST", "localhost")
    port: int = int(os.getenv("DATABASE_PORT", "5433"))
    dbname: str = os.getenv("DATABASE_NAME", "umami")
    user: str = os.getenv("DATABASE_USER", "umami")
    password: str = os.getenv("DATABASE_PASSWORD", "umami")
    
    @property
    def url(self) -> str:
        return os.getenv(
            "DATABASE_URL",
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        )
    
    @property
    def async_url(self) -> str:
        return self.url.replace("postgresql://", "postgresql+asyncpg://")


@dataclass
class GPUSettings:
    """GPU configuration"""
    use_gpu: bool = os.getenv("ML_USE_GPU", "true").lower() == "true"
    gpu_id: Optional[int] = (
        int(os.getenv("ML_GPU_ID")) if os.getenv("ML_GPU_ID") else None
    )
    # Tesla P40 has 24GB, RTX 3060 has 12GB
    max_gpu_memory_pct: float = float(os.getenv("ML_MAX_GPU_MEMORY_PCT", "0.80"))


@dataclass
class ModelPaths:
    """Paths for model storage"""
    base_dir: str = os.getenv("ML_MODEL_DIR", str(Path(__file__).parent / "models" / "saved"))
    
    @property
    def next_page_model(self) -> str: return f"{self.base_dir}/next_page_predictor.onnx"
    @property
    def funnel_model(self) -> str: return f"{self.base_dir}/funnel_predictor.json"
    @property
    def session_intent_model(self) -> str: return f"{self.base_dir}/session_intent.onnx"
    @property
    def recommender_model(self) -> str: return f"{self.base_dir}/recommender.onnx"
    @property
    def page_embeddings(self) -> str: return f"{self.base_dir}/page_embeddings.npy"
    @property
    def markov_model(self) -> str: return f"{self.base_dir}/markov_chain.pkl"


@dataclass
class TrainingConfig:
    """Training parameters"""
    batch_size: int = int(os.getenv("ML_BATCH_SIZE", "256"))
    epochs: int = int(os.getenv("ML_EPOCHS", "50"))
    learning_rate: float = float(os.getenv("ML_LEARNING_RATE", "0.001"))
    max_sequence_length: int = int(os.getenv("ML_MAX_SEQUENCE_LEN", "50"))
    embedding_dim: int = int(os.getenv("ML_EMBEDDING_DIM", "128"))
    hidden_dim: int = int(os.getenv("ML_HIDDEN_DIM", "256"))
    num_layers: int = int(os.getenv("ML_NUM_LAYERS", "4"))


@dataclass
class APIConfig:
    """Service configuration"""
    host: str = os.getenv("ML_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("ML_API_PORT", "8001"))
    log_level: str = os.getenv("ML_LOG_LEVEL", "info")
    # Models to serve on startup
    auto_reload: bool = os.getenv("ML_AUTO_RELOAD", "false").lower() == "true"


@dataclass
class AgeConfig:
    """Apache AGE graph configuration"""
    enabled: bool = os.getenv("AGE_ENABLED", "true").lower() == "true"
    graph_name: str = os.getenv("AGE_GRAPH_NAME", "umami_analytics")
    # AGE uses openCypher for graph queries


@dataclass
class UmamiConfig:
    """Umami app integration"""
    api_url: str = os.getenv("UMAMI_API_URL", "http://localhost:3000")
    api_key: Optional[str] = os.getenv("UMAMI_API_KEY", None)


@dataclass
class MLConfig:
    """Master configuration"""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    gpu: GPUSettings = field(default_factory=GPUSettings)
    paths: ModelPaths = field(default_factory=ModelPaths)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    age: AgeConfig = field(default_factory=AgeConfig)
    umami: UmamiConfig = field(default_factory=UmamiConfig)


# Singleton
CONFIG = MLConfig()
