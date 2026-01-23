# Syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim-bookworm AS production

# Create non-root user
ARG UID=10001
ARG GID=10001
RUN groupadd --gid $GID --system openrag && \
    useradd --uid $UID --gid $GID --system --create-home openrag

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create directories
RUN mkdir -p /app/data /app/logs && \
    chown -R openrag:openrag /app

# Copy application
COPY --chown=openrag:openrag . .

# Switch to non-root user
USER openrag

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENRAG_CONFIG=/app/configs/config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "openrag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
