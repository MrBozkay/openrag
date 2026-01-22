"""Sentence Transformers embedding implementation."""

import logging

from sentence_transformers import SentenceTransformer

from openrag.config import EmbeddingConfig
from openrag.core.base import Embedding

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(Embedding):
    """Sentence Transformers embedding implementation."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        self.config = config
        logger.info(f"Loading embedding model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded model with dimension: {self._dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text documents

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self._dimension
