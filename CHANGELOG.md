# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-19

### Added
- Initial release of OpenRAG framework
- Core RAG pipeline with retrieval and generation
- Vector store implementations:
  - Qdrant vector store with full CRUD operations
  - Chroma vector store for local development
- Embedding support via Sentence Transformers
- LLM integrations:
  - OpenAI (GPT-3.5, GPT-4) with retry logic
  - HuggingFace with quantization support (4-bit, 8-bit)
- Document loaders for TXT, PDF, DOCX, and Markdown files
- Chunking strategies:
  - Fixed-size chunking with overlap
  - Semantic chunking based on paragraphs
- CLI interface with commands:
  - `openrag init` - Project scaffolding
  - `openrag ingest` - Document ingestion
  - `openrag search` - Direct search
  - `openrag serve` - API server
- REST API with FastAPI:
  - `/health` - Health check
  - `/v1/generate` - RAG generation
  - `/v1/generate/stream` - Streaming generation
  - `/v1/search` - Document search
  - `/v1/ingest` - Document ingestion
- Configuration management:
  - YAML-based configuration
  - Environment variable overrides
  - Pydantic validation
- Docker deployment:
  - Dockerfile for containerization
  - Docker Compose with Qdrant and Redis
- Kubernetes deployment:
  - Deployment manifests
  - Service definitions
  - Ingress configuration
  - HorizontalPodAutoscaler
  - ConfigMap and Secret management
- Comprehensive documentation:
  - README with quick start
  - Configuration examples
  - API documentation
  - Architecture overview
- Testing infrastructure:
  - Pytest configuration
  - Unit tests for core components
  - Test fixtures
- Development tools:
  - Ruff for linting
  - Mypy for type checking
  - Black for formatting
  - Pre-configured pyproject.toml

### Technical Details
- Python 3.10+ support
- Full type hints throughout codebase
- Async/await for I/O operations
- Rich console output for CLI
- OpenAPI documentation
- CORS support for web integration
- Health checks for all services
- Graceful error handling
- Structured logging

### Documentation
- Quick start guide
- Installation instructions
- Configuration reference
- Example code
- Docker deployment guide
- Kubernetes deployment guide
- Architecture documentation

## [Unreleased]

### Planned for v2.0
- Hybrid search (vector + keyword)
- Reranking support
- Query caching with Redis
- Built-in evaluation framework
- Prompt template versioning
- Advanced metrics and monitoring

### Planned for v3.0
- Multi-agent orchestration
- Agent memory management
- Cloud provider integrations (AWS, GCP, Azure)
- Advanced observability
- Production monitoring dashboards
