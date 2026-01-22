"""Qdrant vector store implementation."""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from openrag.config import QdrantConfig
from openrag.core.base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize Qdrant client.

        Args:
            config: Qdrant configuration
        """
        self.config = config
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key,
        )
        logger.info(f"Connected to Qdrant at {config.host}:{config.port}")

    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        if await self.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map[self.config.distance_metric],
            ),
        )
        logger.info(
            f"Created collection {collection_name} with vector size {vector_size}"
        )

    async def upsert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert vectors with payloads.

        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            payloads: List of metadata payloads
        """
        points = [
            PointStruct(id=i, vector=vector, payload=payload)
            for i, (vector, payload) in enumerate(zip(vectors, payloads, strict=True))
        ]

        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} vectors to {collection_name}")

    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results
        """
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        search_results = []
        for result in results:
            payload = result.payload or {}
            document = Document(
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
                id=str(result.id),
            )
            search_results.append(
                SearchResult(
                    document=document,
                    score=result.score,
                    chunk_index=payload.get("chunk_index"),
                )
            )

        logger.debug(f"Found {len(search_results)} results for query")
        return search_results

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of the collection
        """
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection {collection_name}")

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        collections = self.client.get_collections().collections
        return any(col.name == collection_name for col in collections)

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection information
        """
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
