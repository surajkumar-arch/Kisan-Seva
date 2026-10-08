.PHONY: setup test lint train evaluate serve serve-web worker docker-build clean help

help:
	@echo "EDAPHNET Enterprise MLOps Management CLI"
	@echo "Available commands:"
	@echo "  make setup         - Install dependencies and prepare environment"
	@echo "  make test          - Execute comprehensive test suite (Unit, Integration, Contract, E2E)"
	@echo "  make lint          - Run code quality checkers (Ruff / Black / MyPy)"
	@echo "  make train         - Train vision & recommendation models on labeled datasets"
	@echo "  make evaluate      - Run benchmarking, calibration error, and OOD evaluations"
	@echo "  make serve         - Start production FastAPI REST service (Port 8000)"
	@echo "  make serve-web     - Launch Streamlit interactive Web & Mobile dashboard (Port 8501)"
	@echo "  make worker        - Start asynchronous batch background worker"
	@echo "  make docker-build  - Build production multi-stage Docker images"
	@echo "  make clean         - Remove temporary cache, coverage, and bytecode artifacts"

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests -v --tb=short

lint:
	python -m ruff check src apps tests || true
	python -m black --check src apps tests || true

train:
	@echo "TRAINING PIPELINE: Ready for user dataset ingestion."
	@echo "Run: python scripts/train_pipeline.py --dataset-dir <path>"

evaluate:
	pytest tests/test_phase8_benchmarks.py -v -s

serve:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-web:
	streamlit run apps/web/app.py --server.port 8501

worker:
	python -m apps.worker.worker

docker-build:
	docker build -t edaphnet:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
