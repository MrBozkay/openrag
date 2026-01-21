"""CLI interface for OpenRAG."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from openrag import __version__
from openrag.chunking import get_chunker
from openrag.config import OpenRAGConfig, VectorStoreType
from openrag.core.ingestion import IngestionPipeline
from openrag.core.pipeline import RAGPipeline
from openrag.core.retriever import Retriever
from openrag.embeddings import SentenceTransformerEmbedding
from openrag.llms import HuggingFaceLLM, OllamaLLM, OpenAILLM
from openrag.vector_stores import ChromaVectorStore, QdrantVectorStore

console = Console()


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """OpenRAG - Enterprise-grade modular RAG framework."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("project_name", default="my-rag-project")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def init(project_name: str, output: Optional[str]) -> None:
    """Initialize a new OpenRAG project."""
    output_dir = Path(output) if output else Path.cwd() / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (output_dir / "src").mkdir(exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "docs").mkdir(exist_ok=True)

    # Create default config
    config = OpenRAGConfig()
    config_path = output_dir / "configs" / "config.yaml"
    config.to_yaml(config_path)

    # Create README
    readme_content = f"""# {project_name}

OpenRAG project initialized with default configuration.

## Quick Start

1. Configure your settings in `configs/config.yaml`
2. Add documents to `data/` directory
3. Run ingestion: `openrag ingest --input ./data`
4. Start API server: `openrag serve`

## Documentation

See https://github.com/openrag/openrag for full documentation.
"""
    (output_dir / "README.md").write_text(readme_content)

    # Create .gitignore
    gitignore_content = """__pycache__/
*.pyc
.env
logs/
chroma_db/
qdrant_storage/
"""
    (output_dir / ".gitignore").write_text(gitignore_content)

    console.print(
        Panel(
            f"[green]✓[/green] Project initialized at: {output_dir}\n\n"
            f"Next steps:\n"
            f"1. cd {output_dir}\n"
            f"2. Edit configs/config.yaml\n"
            f"3. openrag ingest --input ./data",
            title="OpenRAG Project Created",
            border_style="green",
        )
    )


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input directory or file",
)
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--collection", default="openrag", help="Collection name")
@click.pass_context
def ingest(ctx: click.Context, input_path: str, config: Optional[str], collection: str) -> None:
    """Ingest documents into vector store."""
    # Load config
    if config:
        cfg = OpenRAGConfig.from_yaml(Path(config))
    else:
        cfg = OpenRAGConfig()

    async def run_ingestion() -> None:
        # Initialize components
        embedding = SentenceTransformerEmbedding(cfg.embedding)

        if cfg.vector_store.type == VectorStoreType.QDRANT:
            vector_store = QdrantVectorStore(cfg.vector_store.qdrant)
        else:
            vector_store = ChromaVectorStore(cfg.vector_store.chroma)

        pipeline = IngestionPipeline(
            vector_store=vector_store,
            embedding=embedding,
            chunking_config=cfg.chunking,
            collection_name=collection,
        )

        # Ingest
        input_p = Path(input_path)
        if input_p.is_dir():
            stats = await pipeline.ingest_directory(input_p, show_progress=True)
        else:
            from openrag.loaders import DocumentLoader

            doc = DocumentLoader.load(input_p)
            stats = await pipeline.ingest_documents([doc], show_progress=True)

        # Display stats
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Documents", str(stats["documents"]))
        table.add_row("Chunks", str(stats["chunks"]))
        table.add_row("Vectors", str(stats["vectors"]))
        console.print(table)

    asyncio.run(run_ingestion())


@cli.command()
@click.argument("query")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--collection", default="openrag", help="Collection name")
@click.option("--top-k", "-k", type=int, help="Number of results")
@click.pass_context
def search(
    ctx: click.Context, query: str, config: Optional[str], collection: str, top_k: Optional[int]
) -> None:
    """Search for documents."""
    # Load config
    if config:
        cfg = OpenRAGConfig.from_yaml(Path(config))
    else:
        cfg = OpenRAGConfig()

    async def run_search() -> None:
        # Initialize components
        embedding = SentenceTransformerEmbedding(cfg.embedding)

        if cfg.vector_store.type == VectorStoreType.QDRANT:
            vector_store = QdrantVectorStore(cfg.vector_store.qdrant)
        else:
            vector_store = ChromaVectorStore(cfg.vector_store.chroma)

        retriever = Retriever(
            vector_store=vector_store,
            embedding=embedding,
            config=cfg.retrieval,
            collection_name=collection,
        )

        # Search
        results = await retriever.retrieve(query, top_k=top_k)

        # Display results
        for i, result in enumerate(results, 1):
            console.print(
                Panel(
                    f"[bold]Score:[/bold] {result.score:.4f}\n\n"
                    f"{result.document.content[:300]}...\n\n"
                    f"[dim]Source: {result.document.metadata.get('source', 'N/A')}[/dim]",
                    title=f"Result {i}",
                    border_style="blue",
                )
            )

    asyncio.run(run_search())


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--host", help="API host")
@click.option("--port", type=int, help="API port")
@click.pass_context
def serve(
    ctx: click.Context, config: Optional[str], host: Optional[str], port: Optional[int]
) -> None:
    """Start API server."""
    import uvicorn

    # Load config
    if config:
        cfg = OpenRAGConfig.from_yaml(Path(config))
    else:
        cfg = OpenRAGConfig()

    # Override with CLI args
    if host:
        cfg.api.host = host
    if port:
        cfg.api.port = port

    console.print(
        Panel(
            f"[green]Starting OpenRAG API server[/green]\n\n"
            f"Host: {cfg.api.host}\n"
            f"Port: {cfg.api.port}\n"
            f"Docs: http://{cfg.api.host}:{cfg.api.port}/docs",
            title="OpenRAG Server",
            border_style="green",
        )
    )

    uvicorn.run(
        "openrag.api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
        workers=cfg.api.workers,
    )


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
