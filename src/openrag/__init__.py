"""OpenRAG - Enterprise-grade modular RAG framework."""

__version__ = "0.1.0"

from openrag.core.pipeline import RAGPipeline
from openrag.core.retriever import Retriever

__all__ = ["RAGPipeline", "Retriever", "__version__"]
