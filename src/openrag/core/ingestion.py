"""Document ingestion pipeline."""

import logging
from pathlib import Path
from typing import List

from rich.progress import Progress, SpinnerColumn, TextColumn

from openrag.chunking import get_chunker
from openrag.config import ChunkingConfig
from openrag.core.base import Chunk, Document, Embedding, VectorStore
from openrag.loaders import DocumentLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents into vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding: Embedding,
        chunking_config: ChunkingConfig,
        collection_name: str = "openrag",
    ) -> None:
        """Initialize ingestion pipeline.

        Args:
            vector_store: Vector store instance
            embedding: Embedding model instance
            chunking_config: Chunking configuration
            collection_name: Name of the collection
        """
        self.vector_store = vector_store
        self.embedding = embedding
        self.chunking_config = chunking_config
        self.collection_name = collection_name
        self.chunker = get_chunker(chunking_config)

    async def ingest_documents(
        self, documents: List[Document], show_progress: bool = True
    ) -> Dict[str, int]:
        """Ingest documents into vector store.

        Args:
            documents: List of documents to ingest
            show_progress: Whether to show progress bar

        Returns:
            Ingestion statistics
        """
        # Ensure collection exists
        if not await self.vector_store.collection_exists(self.collection_name):
            await self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=self.embedding.dimension,
            )

        all_chunks: List[Chunk] = []

        # Chunk documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            if show_progress:
                task = progress.add_task("Chunking documents...", total=len(documents))

            for doc in documents:
                chunks = self.chunker.chunk(doc)
                all_chunks.extend(chunks)
                if show_progress:
                    progress.update(task, advance=1)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Embed chunks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            if show_progress:
                task = progress.add_task("Embedding chunks...", total=None)

            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding.embed_documents(chunk_texts)

            if show_progress:
                progress.update(task, completed=True)

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Prepare payloads
        payloads = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
            }
            for chunk in all_chunks
        ]

        # Upsert to vector store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            if show_progress:
                task = progress.add_task("Upserting to vector store...", total=None)

            await self.vector_store.upsert(
                collection_name=self.collection_name,
                vectors=embeddings,
                payloads=payloads,
            )

            if show_progress:
                progress.update(task, completed=True)

        stats = {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "vectors": len(embeddings),
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    async def ingest_directory(
        self, directory: Path, recursive: bool = True, show_progress: bool = True
    ) -> Dict[str, int]:
        """Ingest all documents from a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            show_progress: Whether to show progress bar

        Returns:
            Ingestion statistics
        """
        logger.info(f"Loading documents from {directory}")
        documents = DocumentLoader.load_directory(directory, recursive=recursive)

        if not documents:
            logger.warning(f"No documents found in {directory}")
            return {"documents": 0, "chunks": 0, "vectors": 0}

        return await self.ingest_documents(documents, show_progress=show_progress)


from typing import Dict
