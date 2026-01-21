"""Configuration models using Pydantic for validation."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreType(str, Enum):
    """Supported vector store types."""

    QDRANT = "qdrant"
    CHROMA = "chroma"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class QdrantConfig(BaseModel):
    """Qdrant vector store configuration."""

    host: str = Field(default="localhost", description="Qdrant server host")
    port: int = Field(default=6333, description="Qdrant server port")
    collection_name: str = Field(default="openrag", description="Collection name")
    vector_size: int = Field(default=384, description="Vector dimension size")
    distance_metric: Literal["cosine", "euclid", "dot"] = Field(
        default="cosine", description="Distance metric"
    )
    api_key: Optional[str] = Field(default=None, description="API key for cloud Qdrant")


class ChromaConfig(BaseModel):
    """Chroma vector store configuration."""

    persist_directory: str = Field(
        default="./chroma_db", description="Directory for persistent storage"
    )
    collection_name: str = Field(default="openrag", description="Collection name")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    type: VectorStoreType = Field(default=VectorStoreType.QDRANT)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name",
    )
    batch_size: int = Field(default=32, description="Batch size for embedding")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu", description="Device to use")
    normalize_embeddings: bool = Field(
        default=True, description="Normalize embeddings to unit length"
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v


class OpenAIConfig(BaseModel):
    """OpenAI LLM configuration."""

    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class HuggingFaceConfig(BaseModel):
    """HuggingFace LLM configuration."""

    model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf", description="HuggingFace model name"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    load_in_8bit: bool = Field(default=False, description="Load model in 8-bit quantization")
    load_in_4bit: bool = Field(default=False, description="Load model in 4-bit quantization")
    max_new_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    strategy: Literal["fixed", "semantic"] = Field(
        default="fixed", description="Chunking strategy"
    )
    chunk_size: int = Field(default=500, gt=0, description="Chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        """Validate overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = Field(default=5, gt=0, description="Number of documents to retrieve")
    min_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, gt=0, lt=65536, description="API port")
    reload: bool = Field(default=False, description="Enable auto-reload")
    workers: int = Field(default=1, gt=0, description="Number of worker processes")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Optional[str] = Field(default=None, description="Log file path")


class OpenRAGConfig(BaseSettings):
    """Main OpenRAG configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENRAG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "OpenRAGConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
