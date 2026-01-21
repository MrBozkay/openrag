"""Chroma vector store implementation."""

import logging
from typing import Any, Dict, List
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from openrag.config import ChromaConfig
from openrag.core.base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Chroma vector store implementation."""

    def __init__(self, config: ChromaConfig) -> None:
        """Initialize Chroma client.

        Args:
            config: Chroma configuration
        """
        self.config = config
        self.client = chromadb.Client(
            Settings(
                persist_directory=config.persist_directory,
                anonymized_telemetry=False,
            )
        )
        logger.info(f"Initialized Chroma with persist directory: {config.persist_directory}")

    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (not used in Chroma, auto-detected)
        """
        if await self.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        self.client.create_collection(
            name=collection_name,
            metadata={"vector_size": vector_size},
        )
        logger.info(f"Created collection {collection_name}")

    async def upsert(
        self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors with payloads.

        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            payloads: List of metadata payloads
        """
        collection = self.client.get_collection(name=collection_name)

        # Generate IDs
        ids = [str(uuid4()) for _ in range(len(vectors))]

        # Extract documents and metadata
        documents = [payload.get("content", "") for payload in payloads]
        metadatas = [payload.get("metadata", {}) for payload in payloads]

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(vectors)} vectors to {collection_name}")

    async def search(
        self, collection_name: str, query_vector: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results
        """
        collection = self.client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - distance

                document = Document(
                    content=content,
                    metadata=metadata,
                    id=doc_id,
                )
                search_results.append(
                    SearchResult(
                        document=document,
                        score=score,
                        chunk_index=metadata.get("chunk_index"),
                    )
                )

        logger.debug(f"Found {len(search_results)} results for query")
        return search_results

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of the collection
        """
        self.client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection {collection_name}")

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        collections = self.client.list_collections()
        return any(col.name == collection_name for col in collections)

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection information
        """
        collection = self.client.get_collection(name=collection_name)
        count = collection.count()
        metadata = collection.metadata

        return {
            "name": collection_name,
            "count": count,
            "metadata": metadata,
        }
