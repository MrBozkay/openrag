"""Tests for chunking module."""

import pytest

from openrag.chunking import FixedSizeChunker, SemanticChunker, get_chunker
from openrag.config import ChunkingConfig
from openrag.core.base import Document


def test_fixed_size_chunker():
    """Test fixed-size chunking."""
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=20)
    chunker = FixedSizeChunker(config)

    doc = Document(
        content="This is a test document. " * 20,  # ~500 characters
        metadata={"source": "test.txt"},
    )

    chunks = chunker.chunk(doc)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Each chunk should have metadata
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test.txt"

    # Chunks should have content
    for chunk in chunks:
        assert len(chunk.content) > 0


def test_semantic_chunker():
    """Test semantic chunking."""
    config = ChunkingConfig(strategy="semantic", chunk_size=200, chunk_overlap=0)
    chunker = SemanticChunker(config)

    doc = Document(
        content="""Paragraph one is here.
This is still paragraph one.

Paragraph two starts here.
More content in paragraph two.

Paragraph three is the last one.
Final sentence.""",
        metadata={"source": "test.txt"},
    )

    chunks = chunker.chunk(doc)

    # Should create chunks based on paragraphs
    assert len(chunks) >= 1

    # Each chunk should have metadata
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata


def test_get_chunker_factory():
    """Test chunker factory function."""
    # Fixed chunker
    config = ChunkingConfig(strategy="fixed")
    chunker = get_chunker(config)
    assert isinstance(chunker, FixedSizeChunker)

    # Semantic chunker
    config = ChunkingConfig(strategy="semantic")
    chunker = get_chunker(config)
    assert isinstance(chunker, SemanticChunker)

    # Invalid strategy
    config = ChunkingConfig(strategy="fixed")
    config.strategy = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        get_chunker(config)


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    config = ChunkingConfig(strategy="fixed", chunk_size=50, chunk_overlap=10)
    chunker = FixedSizeChunker(config)

    doc = Document(content="A" * 200, metadata={})
    chunks = chunker.chunk(doc)

    # Should have multiple chunks
    assert len(chunks) > 1
