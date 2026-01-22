# OpenRAG Development Makefile

.PHONY: help install dev install-dev test lint typecheck format security clean build run docker docker-build docker-run docker-compose-up docker-compose-down k8s-deploy k8s-delete

help:
	@echo "OpenRAG Development Commands"
	@echo "make install - Install package"
	@echo "make dev - Install in development mode"
	@echo "make test - Run tests with coverage"
	@echo "make lint - Run Ruff linter"
	@echo "make format - Format code"
	@echo "make typecheck - Run Mypy type checker"
	@echo "make security - Run security scans"
	@echo "make clean - Clean build artifacts"
	@echo "make build - Build Python package"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-compose-up - Start all services"
	@echo "make k8s-deploy - Deploy to Kubernetes"

install:
	pip install --upgrade pip
	pip install -e .

dev: install
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/openrag --cov-report=term-missing --cov-report=html

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .
	black .

typecheck:
	mypy src/

security:
	bandit -r src/ --exclude tests/
	pip-audit

pre-commit:
	pre-commit run --all-files

ci: lint typecheck test security
	@echo "CI pipeline completed!"

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t openrag:latest .

docker-build-multi:
	docker buildx build --platform linux/amd64,linux/arm64 -t openrag:latest --push .

docker-run:
	docker run -p 8000:8000 --env-file .env openrag:latest

docker-push:
	docker tag openrag:latest ghcr.io/mrbozkay/openrag:latest
	docker push ghcr.io/mrbozkay/openrag:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

k8s-deploy:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

k8s-logs:
	kubectl logs -f deployment/openrag

release: clean build test
	python -m build

docs:
	sphinx-apidoc -o docs/ src/openrag/
	cd docs && make html

version:
	python -c "import openrag; print(openrag.__version__)"

deps:
	pip list --outdated
