"""RAG pipeline for generation with retrieval."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from openrag.core.base import LLM, SearchResult
from openrag.core.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGResponse:
    """RAG response with generated text and sources."""

    def __init__(self, text: str, sources: list[SearchResult]) -> None:
        """Initialize RAG response.

        Args:
            text: Generated text
            sources: Source documents used
        """
        self.text = text
        self.sources = sources

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "sources": [
                {
                    "content": source.document.content[:200] + "...",
                    "score": source.score,
                    "metadata": source.document.metadata,
                }
                for source in self.sources
            ],
        }


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the sources you use in your answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize RAG pipeline.

        Args:
            retriever: Retriever instance
            llm: LLM instance
            system_prompt: Optional custom system prompt
        """
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        top_k: int | None = None,
        include_sources: bool = True,
        **llm_kwargs: Any,
    ) -> RAGResponse:
        """Generate response using RAG.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            include_sources: Whether to include source citations
            **llm_kwargs: Additional LLM parameters

        Returns:
            RAG response with text and sources
        """
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        sources = await self.retriever.retrieve(query, top_k=top_k)

        if not sources:
            logger.warning("No relevant documents found")
            response_text = (
                "I couldn't find any relevant information to answer your question."
            )
            return RAGResponse(text=response_text, sources=[])

        # Construct context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            source_info = f"Source {i}"
            if source.document.metadata.get("source"):
                source_info += f" ({source.document.metadata['source']})"
            context_parts.append(f"{source_info}:\n{source.document.content}\n")

        context = "\n".join(context_parts)

        # Construct prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        logger.info("Generating response...")
        response_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            **llm_kwargs,
        )

        logger.info(f"Generated response: {len(response_text)} characters")
        return RAGResponse(
            text=response_text, sources=sources if include_sources else []
        )

    async def generate_stream(
        self,
        query: str,
        top_k: int | None = None,
        **llm_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate response with streaming.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            **llm_kwargs: Additional LLM parameters

        Yields:
            Text chunks
        """
        # Retrieve relevant documents
        sources = await self.retriever.retrieve(query, top_k=top_k)

        if not sources:
            yield "I couldn't find any relevant information to answer your question."
            return

        # Construct context
        context_parts = []
        for i, source in enumerate(sources, 1):
            source_info = f"Source {i}"
            if source.document.metadata.get("source"):
                source_info += f" ({source.document.metadata['source']})"
            context_parts.append(f"{source_info}:\n{source.document.content}\n")

        context = "\n".join(context_parts)

        # Construct prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Stream response
        async for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self.system_prompt,
            **llm_kwargs,
        ):
            yield chunk
