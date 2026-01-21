"""FastAPI application for OpenRAG."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from openrag import __version__
from openrag.config import OpenRAGConfig, VectorStoreType
from openrag.core.ingestion import IngestionPipeline
from openrag.core.pipeline import RAGPipeline
from openrag.core.retriever import Retriever
from openrag.embeddings import SentenceTransformerEmbedding
from openrag.llms import HuggingFaceLLM, OllamaLLM, OpenAILLM
from openrag.loaders import DocumentLoader
from openrag.vector_stores import ChromaVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)

# Global state
config: OpenRAGConfig
rag_pipeline: RAGPipeline
ingestion_pipeline: IngestionPipeline
retriever: Retriever


@asynccontextmanager
def lifespan(app: FastAPI) -> Any:
    """Lifespan context manager for startup/shutdown."""
    global config, rag_pipeline, ingestion_pipeline, retriever

    # Load configuration
    try:
        config = OpenRAGConfig.from_yaml(Path("configs/config.yaml"))
    except FileNotFoundError:
        config = OpenRAGConfig()

    logger.info("Initializing OpenRAG components...")

    # Initialize embedding
    embedding = SentenceTransformerEmbedding(config.embedding)

    # Initialize vector store
    if config.vector_store.type == VectorStoreType.QDRANT:
        vector_store = QdrantVectorStore(config.vector_store.qdrant)
    else:
        vector_store = ChromaVectorStore(config.vector_store.chroma)

    # Initialize LLM
    if config.llm.provider.value == "openai":
        llm = OpenAILLM(config.llm.openai)
    elif config.llm.provider.value == "ollama":
        llm = OllamaLLM(config.llm.ollama)
    else:
        llm = HuggingFaceLLM(config.llm.huggingface)

    # Initialize pipelines
    retriever = Retriever(
        vector_store=vector_store,
        embedding=embedding,
        config=config.retrieval,
    )

    rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)

    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding=embedding,
        chunking_config=config.chunking,
    )

    logger.info("OpenRAG initialized successfully")

    try:
        yield
    finally:
        logger.info("Shutting down OpenRAG")


app = FastAPI(
    title="OpenRAG API",
    description="Enterprise-grade modular RAG framework",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for generation."""

    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source citations")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)


class GenerateResponse(BaseModel):
    """Response model for generation."""

    text: str = Field(..., description="Generated text")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(None, description="Number of results")


class SearchResponse(BaseModel):
    """Response model for search."""

    results: List[Dict[str, Any]] = Field(..., description="Search results")


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    documents: int = Field(..., description="Number of documents ingested")
    chunks: int = Field(..., description="Number of chunks created")
    vectors: int = Field(..., description="Number of vectors stored")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate response using RAG.

    Args:
        request: Generation request

    Returns:
        Generated response with sources
    """
    try:
        llm_kwargs = {}
        if request.temperature is not None:
            llm_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            llm_kwargs["max_tokens"] = request.max_tokens

        response = await rag_pipeline.generate(
            query=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources,
            **llm_kwargs,
        )

        return GenerateResponse(
            text=response.text,
            sources=[
                {
                    "content": source.document.content[:200] + "...",
                    "score": source.score,
                    "metadata": source.document.metadata,
                }
                for source in response.sources
            ],
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate/stream")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Generate response with streaming.

    Args:
        request: Generation request

    Returns:
        Streaming response
    """

    async def generate_stream_response():
        try:
            llm_kwargs = {}
            if request.temperature is not None:
                llm_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                llm_kwargs["max_tokens"] = request.max_tokens

            agen = rag_pipeline.generate_stream(
                query=request.query,
                top_k=request.top_k,
                **llm_kwargs,
            )
            async for chunk in agen:
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_stream_response(), media_type="text/plain")


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search for relevant documents.

    Args:
        request: Search request

    Returns:
        Search results
    """
    try:
        results = await retriever.retrieve(request.query, top_k=request.top_k)

        return SearchResponse(
            results=[
                {
                    "content": result.document.content,
                    "score": result.score,
                    "metadata": result.document.metadata,
                }
                for result in results
            ]
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    """Ingest documents.

    Args:
        files: List of files to ingest

    Returns:
        Ingestion statistics
    """
    try:
        import tempfile

        documents = []

        # Save uploaded files temporarily and load
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in files:
                if file.filename is None:
                    continue
                file_path = Path(tmpdir) / file.filename
                content = await file.read()
                file_path.write_bytes(content)

                try:
                    doc = DocumentLoader.load(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file.filename}: {e}")

        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents provided")

        stats = await ingestion_pipeline.ingest_documents(documents, show_progress=False)

        return IngestResponse(
            documents=stats["documents"],
            chunks=stats["chunks"],
            vectors=stats["vectors"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
