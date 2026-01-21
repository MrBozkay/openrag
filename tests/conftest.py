"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    from openrag.core.base import Document

    return Document(
        content="This is a sample document for testing. It contains multiple sentences. "
        "The content is used to test various components of the OpenRAG framework.",
        metadata={"source": "test.txt", "type": "txt"},
        id="test-doc-1",
    )


@pytest.fixture
def sample_documents():
    """Multiple sample documents for testing."""
    from openrag.core.base import Document

    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml.txt"},
            id="doc-1",
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "dl.txt"},
            id="doc-2",
        ),
        Document(
            content="Natural language processing enables computers to understand human language.",
            metadata={"source": "nlp.txt"},
            id="doc-3",
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from openrag.core.base import Chunk

    return [
        Chunk(
            content="Machine learning is a subset of AI.",
            metadata={"chunk_index": 0, "source": "ml.txt"},
        ),
        Chunk(
            content="Deep learning uses neural networks.",
            metadata={"chunk_index": 0, "source": "dl.txt"},
        ),
    ]
