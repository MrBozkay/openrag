# Contributing to OpenRAG

Thank you for your interest in contributing to OpenRAG! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

For feature requests:
- Check existing issues first
- Provide clear use case
- Explain why this feature would be valuable
- Consider implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

4. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   
   # Lint
   ruff check src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest --cov=src/openrag
   ```

5. **Commit your changes**
   ```bash
   git commit -m "feat: add new feature"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style

### Python Style
- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings (Google style)

### Example:
```python
def process_document(
    document: Document,
    config: ChunkingConfig,
) -> List[Chunk]:
    """Process a document into chunks.
    
    Args:
        document: Input document to process
        config: Chunking configuration
        
    Returns:
        List of document chunks
        
    Raises:
        ValueError: If document is empty
    """
    if not document.content:
        raise ValueError("Document content cannot be empty")
    
    # Implementation
    return chunks
```

### Docstring Format
```python
"""Short description.

Longer description if needed.

Args:
    param1: Description of param1
    param2: Description of param2

Returns:
    Description of return value

Raises:
    ExceptionType: When this exception is raised
"""
```

## 🧪 Testing

### Writing Tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures from `conftest.py`
- Aim for >80% code coverage

### Example Test:
```python
import pytest
from openrag.core.base import Document
from openrag.chunking import FixedSizeChunker

def test_fixed_size_chunker(sample_document):
    """Test fixed-size chunking."""
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk(sample_document)
    
    assert len(chunks) > 0
    assert all(chunk.content for chunk in chunks)
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src/openrag --cov-report=html

# Specific test file
pytest tests/test_chunking.py

# Specific test
pytest tests/test_chunking.py::test_fixed_size_chunker
```

## 📚 Documentation

### Code Documentation
- All public functions must have docstrings
- Include type hints
- Provide usage examples for complex functions

### README Updates
- Update README.md for new features
- Add examples to `examples/` directory
- Update CHANGELOG.md

## 🏗️ Architecture Guidelines

### Adding New Components

#### New Vector Store
1. Create file in `src/openrag/vector_stores/`
2. Inherit from `VectorStore` base class
3. Implement all abstract methods
4. Add configuration to `config.py`
5. Add tests
6. Update documentation

#### New LLM Provider
1. Create file in `src/openrag/llms/`
2. Inherit from `LLM` base class
3. Implement `generate()` and `generate_stream()`
4. Add configuration
5. Add tests
6. Update documentation

#### New Chunking Strategy
1. Add to `src/openrag/chunking/chunkers.py`
2. Inherit from `Chunker` base class
3. Implement `chunk()` method
4. Update `get_chunker()` factory
5. Add tests

## 🔍 Code Review Process

### What We Look For
- ✅ Code follows style guidelines
- ✅ Tests are included and passing
- ✅ Documentation is updated
- ✅ Type hints are present
- ✅ No breaking changes (or clearly documented)
- ✅ Commit messages follow conventions

### Review Timeline
- Initial review within 3 days
- Feedback addressed within 1 week
- Merge after approval from 2 maintainers

## 🐛 Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### CLI Debug Mode
```bash
openrag --verbose ingest --input ./data
```

### Common Issues
1. **Import errors**: Check virtual environment activation
2. **API key errors**: Verify environment variables
3. **Vector store connection**: Ensure Qdrant/Chroma is running

## 📦 Release Process

### Version Numbering
- Follow Semantic Versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test Docker image
5. Create GitHub release
6. Publish to PyPI

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📧 Contact

- GitHub Issues: Primary communication channel
- Discussions: For questions and ideas
- Email: maintainers@openrag.dev

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to OpenRAG! 🚀
