"""Document chunking implementations."""

import logging
import re

from openrag.config import ChunkingConfig
from openrag.core.base import Chunk, Chunker, Document

logger = logging.getLogger(__name__)


class FixedSizeChunker(Chunker):
    """Fixed-size chunking with overlap."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        text = document.content
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.config.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > len(chunk_text) // 2:  # Only if in second half
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + len(chunk_text)

            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "source_id": document.id,
            }

            chunks.append(Chunk(content=chunk_text.strip(), metadata=metadata))

            start = end - self.config.chunk_overlap
            chunk_index += 1

        logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks


class SemanticChunker(Chunker):
    """Semantic chunking based on paragraph/section boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into semantic chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        text = document.content

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, save current chunk
            if (
                current_chunk
                and len(current_chunk) + len(para) > self.config.chunk_size
            ):
                metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "source_id": document.id,
                }
                chunks.append(Chunk(content=current_chunk.strip(), metadata=metadata))
                current_chunk = para
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add remaining chunk
        if current_chunk:
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "source_id": document.id,
            }
            chunks.append(Chunk(content=current_chunk.strip(), metadata=metadata))

        logger.debug(f"Created {len(chunks)} semantic chunks from document")
        return chunks


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Factory function to get chunker based on config.

    Args:
        config: Chunking configuration

    Returns:
        Chunker instance
    """
    if config.strategy == "fixed":
        return FixedSizeChunker(config)
    elif config.strategy == "semantic":
        return SemanticChunker(config)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")
