# OpenRAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/openrag/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/openrag/actions/workflows/ci.yml)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

**OpenRAG** is an enterprise-grade, modular, and customizable open-source RAG (Retrieval-Augmented Generation) framework. It abstracts the complexity of existing RAG libraries (LangChain, LlamaIndex) to help full-stack developers quickly build RAG-based applications.

## ✨ Features

- 🚀 **Quick Setup**: Get a working RAG pipeline in 5 minutes
- 🔧 **Modular Architecture**: Swap components independently
- 🗄️ **Multiple Vector Stores**: Qdrant and Chroma support
- 🤖 **Multiple LLM Providers**: OpenAI and HuggingFace
- 🌐 **REST API & CLI**: Easy integration options
- 🐳 **Docker & Kubernetes**: Production-ready deployment
- 📊 **Type-Safe**: Full type hints and Pydantic validation
- 🧪 **Well-Tested**: Comprehensive test coverage

## 🚀 Quick Start

### Installation

```bash
pip install openrag
```

Or install from source:

```bash
git clone https://github.com/yourusername/openrag.git
cd openrag
pip install -e .
```

### Initialize a Project

```bash
openrag init my-rag-project
cd my-rag-project
```

### Configure

Edit `configs/config.yaml` to set your API keys and preferences:

```yaml
llm:
  provider: openai
  openai:
    api_key: your-api-key-here
```

Or set via environment variable:

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Ingest Documents

```bash
# Add your documents to the data/ directory
openrag ingest --input ./data
```

### Start API Server

```bash
openrag serve
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Query via CLI

```bash
openrag search "What is RAG?"
```

## 📖 Usage Examples

### Python API

```python
import asyncio
from openrag import RAGPipeline
from openrag.config import OpenRAGConfig
from openrag.core.retriever import Retriever
from openrag.embeddings import SentenceTransformerEmbedding
from openrag.llms import OpenAILLM
from openrag.vector_stores import QdrantVectorStore

async def main():
    # Load configuration
    config = OpenRAGConfig.from_yaml("configs/config.yaml")
    
    # Initialize components
    embedding = SentenceTransformerEmbedding(config.embedding)
    vector_store = QdrantVectorStore(config.vector_store.qdrant)
    llm = OpenAILLM(config.llm.openai)
    
    # Create retriever and pipeline
    retriever = Retriever(vector_store, embedding, config.retrieval)
    pipeline = RAGPipeline(retriever, llm)
    
    # Generate response
    response = await pipeline.generate("What is RAG?")
    print(response.text)
    
    # Print sources
    for i, source in enumerate(response.sources, 1):
        print(f"\nSource {i}: {source.document.metadata.get('source')}")
        print(f"Score: {source.score:.4f}")

asyncio.run(main())
```

### REST API

```bash
# Generate response
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 5,
    "include_sources": true
  }'

# Search documents
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 3
  }'
```

## 🏗️ Architecture

OpenRAG follows a modular plugin-based architecture:

```
┌─────────────────────────────────────────┐
│           RAG Pipeline                  │
├─────────────────────────────────────────┤
│  Retriever  │  LLM  │  Chunker         │
├─────────────────────────────────────────┤
│  Vector Store  │  Embedding Model      │
└─────────────────────────────────────────┘
```

### Components

- **Vector Stores**: Qdrant, Chroma
- **Embeddings**: Sentence Transformers
- **LLMs**: OpenAI (GPT-3.5, GPT-4), HuggingFace (Llama, Mistral, etc.)
- **Chunkers**: Fixed-size, Semantic
- **Loaders**: TXT, PDF, DOCX, Markdown

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Set your API key
export OPENAI_API_KEY=your-api-key-here

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f openrag
```

Services included:
- **openrag**: Main API server (port 8000)
- **qdrant**: Vector database (port 6333)
- **redis**: Cache and queue (port 6379)

## ☸️ Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace openrag

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods -n openrag
kubectl get svc -n openrag
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Examples](docs/examples/)
- [Architecture](docs/architecture.md)

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/openrag.git
cd openrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/openrag --cov-report=html

# Run linting
ruff check src/

# Run type checking
mypy src/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## 🗺️ Roadmap

### v1.0 (Current)
- ✅ Core RAG pipeline
- ✅ Qdrant and Chroma support
- ✅ OpenAI and HuggingFace LLMs
- ✅ REST API and CLI
- ✅ Docker and Kubernetes deployment

### v2.0 (Planned)
- 🔄 Hybrid search (vector + keyword)
- 🔄 Reranking support
- 🔄 Query caching with Redis
- 🔄 Built-in evaluation framework
- 🔄 Prompt template versioning

### v3.0 (Future)
- 🔮 Multi-agent orchestration
- 🔮 Agent memory management
- 🔮 Cloud provider integrations (AWS, GCP, Azure)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

For security issues, please see our [Security Policy](SECURITY.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for inspiration
- [Qdrant](https://qdrant.tech/) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📧 Contact

- GitHub Issues: [https://github.com/yourusername/openrag/issues](https://github.com/yourusername/openrag/issues)
- Email: contact@openrag.dev

---

Made with ❤️ by the OpenRAG Team
