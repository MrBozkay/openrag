"""Retriever for searching documents."""

import logging
from typing import List

from openrag.config import RetrievalConfig
from openrag.core.base import Embedding, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Document retriever using vector search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding: Embedding,
        config: RetrievalConfig,
        collection_name: str = "openrag",
    ) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector store instance
            embedding: Embedding model instance
            config: Retrieval configuration
            collection_name: Name of the collection to search
        """
        self.vector_store = vector_store
        self.embedding = embedding
        self.config = config
        self.collection_name = collection_name

    async def retrieve(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return (overrides config)

        Returns:
            List of search results
        """
        # Embed query
        query_vector = self.embedding.embed_query(query)

        # Search vector store
        k = top_k if top_k is not None else self.config.top_k
        results = await self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=k,
        )

        # Filter by minimum similarity
        filtered_results = [
            result for result in results if result.score >= self.config.min_similarity
        ]

        logger.info(
            f"Retrieved {len(filtered_results)}/{len(results)} results "
            f"(min_similarity={self.config.min_similarity})"
        )

        return filtered_results
