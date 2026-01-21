"""Basic RAG example."""

import asyncio
import os
from pathlib import Path

from openrag.chunking import get_chunker
from openrag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    OpenAIConfig,
    QdrantConfig,
    RetrievalConfig,
)
from openrag.core.ingestion import IngestionPipeline
from openrag.core.pipeline import RAGPipeline
from openrag.core.retriever import Retriever
from openrag.embeddings import SentenceTransformerEmbedding
from openrag.llms import OpenAILLM
from openrag.loaders import DocumentLoader
from openrag.vector_stores import QdrantVectorStore


async def main():
    """Run basic RAG example."""
    print("🚀 OpenRAG Basic Example\n")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your-key-here")
        return

    # Configuration
    print("📝 Setting up configuration...")
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
    )
    qdrant_config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="openrag_example",
    )
    openai_config = OpenAIConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=api_key,
    )
    chunking_config = ChunkingConfig(
        strategy="fixed",
        chunk_size=500,
        chunk_overlap=50,
    )
    retrieval_config = RetrievalConfig(top_k=3, min_similarity=0.0)

    # Initialize components
    print("🔧 Initializing components...")
    embedding = SentenceTransformerEmbedding(embedding_config)
    vector_store = QdrantVectorStore(qdrant_config)
    llm = OpenAILLM(openai_config)

    # Create sample documents
    print("\n📄 Creating sample documents...")
    sample_docs_dir = Path("sample_docs")
    sample_docs_dir.mkdir(exist_ok=True)

    # Create sample documents
    (sample_docs_dir / "ml.txt").write_text(
        """Machine Learning Overview

Machine learning is a subset of artificial intelligence that focuses on the development 
of algorithms and statistical models that enable computers to improve their performance 
on a specific task through experience.

There are three main types of machine learning:
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

Common applications include image recognition, natural language processing, 
recommendation systems, and autonomous vehicles."""
    )

    (sample_docs_dir / "rag.txt").write_text(
        """Retrieval-Augmented Generation (RAG)

RAG is a technique that combines information retrieval with text generation. 
It works by first retrieving relevant documents from a knowledge base, then using 
those documents as context for a language model to generate responses.

Benefits of RAG:
- Reduces hallucinations by grounding responses in retrieved facts
- Enables LLMs to access up-to-date information
- More cost-effective than fine-tuning for many use cases
- Provides source attribution for generated content

RAG systems typically consist of:
1. Document ingestion and chunking
2. Vector embedding and storage
3. Similarity search for retrieval
4. Context-aware generation with LLM"""
    )

    # Load and ingest documents
    print("📚 Loading documents...")
    documents = DocumentLoader.load_directory(sample_docs_dir)
    print(f"   Loaded {len(documents)} documents")

    print("\n🔄 Ingesting documents into vector store...")
    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding=embedding,
        chunking_config=chunking_config,
        collection_name="openrag_example",
    )

    stats = await ingestion_pipeline.ingest_documents(documents, show_progress=True)
    print(f"\n   ✅ Ingested {stats['documents']} documents")
    print(f"   ✅ Created {stats['chunks']} chunks")
    print(f"   ✅ Stored {stats['vectors']} vectors")

    # Create RAG pipeline
    print("\n🤖 Creating RAG pipeline...")
    retriever = Retriever(
        vector_store=vector_store,
        embedding=embedding,
        config=retrieval_config,
        collection_name="openrag_example",
    )
    rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)

    # Example queries
    queries = [
        "What is machine learning?",
        "What are the benefits of RAG?",
        "Explain the types of machine learning",
    ]

    print("\n" + "=" * 80)
    print("💬 Running example queries...")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n📌 Query {i}: {query}")
        print("-" * 80)

        response = await rag_pipeline.generate(query, top_k=3)

        print(f"\n🤖 Response:\n{response.text}\n")

        if response.sources:
            print(f"📚 Sources ({len(response.sources)}):")
            for j, source in enumerate(response.sources, 1):
                print(f"\n   {j}. Score: {source.score:.4f}")
                print(f"      Source: {source.document.metadata.get('source', 'N/A')}")
                print(f"      Preview: {source.document.content[:100]}...")

        print("\n" + "=" * 80)

    print("\n✅ Example completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Try your own queries")
    print("   2. Add more documents to sample_docs/")
    print("   3. Experiment with different configurations")
    print("   4. Check out the API with: openrag serve")


if __name__ == "__main__":
    asyncio.run(main())
