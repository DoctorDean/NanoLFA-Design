.PHONY: help check test lint typecheck format setup-data clean

PYTHON := python
CONDA_ENV := nanolfa

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

check: lint typecheck test  ## Run all checks

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src/ scripts/ tests/

format:  ## Auto-format code
	$(PYTHON) -m ruff format src/ scripts/ tests/

typecheck:  ## Type check with mypy
	$(PYTHON) -m mypy src/nanolfa/

# ---------------------------------------------------------------------------
# Data & Model Setup
# ---------------------------------------------------------------------------

setup-data:  ## Download required databases and model weights
	@echo "=== Downloading AlphaFold model parameters ==="
	bash scripts/setup/download_af_params.sh
	@echo "=== Downloading ProteinMPNN weights ==="
	bash scripts/setup/download_proteinmpnn.sh
	@echo "=== Downloading VHH germline templates ==="
	$(PYTHON) scripts/setup/fetch_imgt_germlines.py
	@echo "=== Preparing target molecule structures ==="
	$(PYTHON) scripts/prepare_targets.py --config configs/default.yaml
	@echo "=== Setup complete ==="

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

run-pdg:  ## Run full pipeline for PdG target
	$(PYTHON) scripts/run_pipeline.py --config configs/targets/pdg.yaml

run-e3g:  ## Run full pipeline for E3G target
	$(PYTHON) scripts/run_pipeline.py --config configs/targets/e3g.yaml

run-round:  ## Run a single design round (ROUND=N TARGET=pdg|e3g)
	$(PYTHON) scripts/run_design_round.py \
		--target $(TARGET) \
		--round $(ROUND) \
		--config configs/targets/$(TARGET).yaml

cross-react:  ## Screen cross-reactivity (TARGET=pdg|e3g ROUND=N)
	$(PYTHON) scripts/screen_crossreactivity.py \
		--candidates data/results/round_$(ROUND)/top_candidates.pdb \
		--panel configs/targets/$(TARGET).yaml

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove generated artifacts
	rm -rf data/results/round_*/
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean  ## Remove everything including downloaded data
	rm -rf data/targets/*/structures/
	rm -rf data/templates/germline_vhh/
