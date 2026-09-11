# Every number in the documentation is produced by one of these targets.
PY ?= .venv/bin/python
PORT ?= 8077

.DEFAULT_GOAL := help
.PHONY: help setup data train serve smoke test lint format bench drift retrain rollback \
        results verify clean clean-all all

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install everything
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements-dev.txt
	$(PY) -m pip install --quiet -e . --no-deps
	@echo "SETUP_OK"

data: ## Download and checksum-verify the raw dataset
	$(PY) scripts/fetch_data.py

train: ## Validate, train, evaluate, track in MLflow and register
	$(PY) scripts/train.py

serve: ## Run the serving API
	$(PY) scripts/serve.py --port $(PORT)

smoke: ## Start the API, exercise every endpoint, shut it down
	$(PY) scripts/smoke_test.py

test: ## Run the full test suite
	$(PY) -m pytest -q --durations=15

lint: ## Lint and format-check
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts

format: ## Apply formatting
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m black src tests scripts

bench: ## Run the HTTP load test, pinned (production) and unpinned (comparison)
	$(PY) scripts/load_test.py --thread-pin 1 --duration 20 --warmup 5 \
	      --concurrency 1 2 4 8 16 --batch-size 1 32 --wait-for-quiet-host 600
	$(PY) scripts/load_test.py --thread-pin 0 --duration 20 --warmup 5 \
	      --concurrency 1 8 --batch-size 1 32

drift: ## Run the controlled drift experiment
	$(PY) scripts/drift_experiment.py --trials 30 --latency-trials 10

retrain: ## Run the retraining / promotion / rejection experiment
	$(PY) scripts/retrain_experiment.py

rollback: ## Verify rollback through the running serving layer
	$(PY) scripts/rollback_through_serving.py

results: ## Aggregate every measurement into the top-level CSVs and docs tables
	$(PY) scripts/collect_results.py

verify: ## Re-run every acceptance check
	$(PY) scripts/verify/run_all.py

all: data train smoke test drift retrain rollback bench results ## Full pipeline

clean: ## Remove caches and generated results
	rm -rf .pytest_cache .ruff_cache **/__pycache__ results/serving/*.log
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-all: clean ## Also remove the MLflow store, artefacts and processed data
	rm -rf mlruns mlflow.db artifacts data/processed data/scenarios
