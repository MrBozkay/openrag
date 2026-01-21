"""Base abstractions for OpenRAG components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Document(BaseModel):
    """Document model with content and metadata."""

    content: str
    metadata: Dict[str, Any] = {}
    id: Optional[str] = None


class Chunk(BaseModel):
    """Document chunk with metadata."""

    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class SearchResult(BaseModel):
    """Search result with document and score."""

    document: Document
    score: float
    chunk_index: Optional[int] = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    async def upsert(
        self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors with payloads."""
        pass

    @abstractmethod
    async def search(
        self, collection_name: str, query_vector: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass


class Embedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class LLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Generate text with streaming."""
        pass


class Chunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks."""
        pass
