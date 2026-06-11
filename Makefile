PYTHON := python3
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

.PHONY: install download train test test-unit test-integration test-smoke \
        mlflow-ui api docker-build docker-up docker-mlflow docker-down clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download:
	$(PY) scripts/download_data.py

train: test-unit
	$(PY) scripts/train_models.py

pipeline: test-unit
	$(PY) -m scripts.build_pipeline

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest tests/ -k "not integration and not smoke" -v

test-integration:
	$(PY) -m pytest tests/integration/ -v

test-smoke:
	$(PY) -m pytest tests/smoke/ -v

api:
	ARTIFACTS_DIR=artifacts $(PY) -m uvicorn app.main:app --reload --port 8000

mlflow-ui:
	$(PY) -m mlflow ui \
		--backend-store-uri sqlite:///mlflow/mlflow.db \
		--default-artifact-root mlflow/artifacts \
		--port 5000

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d api log-reader

docker-mlflow:
	docker compose --profile tracking up -d mlflow

docker-train:
	docker compose run --rm train

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

clean:
	rm -rf $(VENV) artifacts/*.pkl reports/figures/*.png .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
