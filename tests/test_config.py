"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from openrag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenRAGConfig,
    QdrantConfig,
    RetrievalConfig,
)


def test_qdrant_config_defaults():
    """Test Qdrant config with defaults."""
    config = QdrantConfig()
    assert config.host == "localhost"
    assert config.port == 6333
    assert config.collection_name == "openrag"
    assert config.vector_size == 384
    assert config.distance_metric == "cosine"


def test_embedding_config_validation():
    """Test embedding config validation."""
    # Valid config
    config = EmbeddingConfig(batch_size=32)
    assert config.batch_size == 32

    # Invalid batch size
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)

    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=-1)


def test_chunking_config_validation():
    """Test chunking config validation."""
    # Valid config
    config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50

    # Invalid: overlap >= chunk_size
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=150)


def test_openai_config_validation():
    """Test OpenAI config validation."""
    # Valid config
    config = OpenAIConfig(temperature=0.7, max_tokens=1000)
    assert config.temperature == 0.7
    assert config.max_tokens == 1000

    # Invalid temperature
    with pytest.raises(ValidationError):
        OpenAIConfig(temperature=3.0)

    # Invalid max_tokens
    with pytest.raises(ValidationError):
        OpenAIConfig(max_tokens=0)


def test_retrieval_config_validation():
    """Test retrieval config validation."""
    # Valid config
    config = RetrievalConfig(top_k=5, min_similarity=0.5)
    assert config.top_k == 5
    assert config.min_similarity == 0.5

    # Invalid top_k
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=0)

    # Invalid min_similarity
    with pytest.raises(ValidationError):
        RetrievalConfig(min_similarity=1.5)


def test_openrag_config_yaml_serialization():
    """Test YAML serialization/deserialization."""
    config = OpenRAGConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Save to YAML
        config.to_yaml(config_path)
        assert config_path.exists()

        # Load from YAML
        loaded_config = OpenRAGConfig.from_yaml(config_path)
        assert loaded_config.vector_store.type == config.vector_store.type
        assert loaded_config.embedding.model_name == config.embedding.model_name
        assert loaded_config.llm.provider == config.llm.provider


def test_openrag_config_env_override():
    """Test environment variable override."""
    import os

    # Set environment variable
    os.environ["OPENRAG_VECTOR_STORE__TYPE"] = "chroma"
    os.environ["OPENRAG_EMBEDDING__BATCH_SIZE"] = "64"

    config = OpenRAGConfig()
    assert config.vector_store.type.value == "chroma"
    assert config.embedding.batch_size == 64

    # Cleanup
    del os.environ["OPENRAG_VECTOR_STORE__TYPE"]
    del os.environ["OPENRAG_EMBEDDING__BATCH_SIZE"]


def test_ollama_config_defaults():
    """Test Ollama config with defaults."""
    config = OllamaConfig()
    assert config.host == "http://localhost"
    assert config.port == 11434
    assert config.model == "llama3"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.timeout == 60


def test_ollama_config_custom_values():
    """Test Ollama config with custom values."""
    config = OllamaConfig(
        host="http://ollama.example.com",
        port=8080,
        model="mistral",
        temperature=0.5,
        max_tokens=512,
        timeout=120,
    )
    assert config.host == "http://ollama.example.com"
    assert config.port == 8080
    assert config.model == "mistral"
    assert config.temperature == 0.5
    assert config.max_tokens == 512
    assert config.timeout == 120


def test_ollama_config_validation():
    """Test Ollama config validation."""
    # Valid config with system prompt
    config = OllamaConfig(system_prompt="You are a helpful assistant.")
    assert config.system_prompt == "You are a helpful assistant."

    # Invalid temperature
    with pytest.raises(ValidationError):
        OllamaConfig(temperature=3.0)

    # Invalid max_tokens
    with pytest.raises(ValidationError):
        OllamaConfig(max_tokens=0)
