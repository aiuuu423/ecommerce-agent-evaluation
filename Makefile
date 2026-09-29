PYTHON ?= python3
DEVELOPMENT_CONFIG ?= configs/data/synthetic_v1.yaml
PUBLIC_VALIDATION_CONFIG ?= configs/data/synthetic_public_validation_v1.yaml
TOOL_CONTRACT ?= configs/evaluation/tool_contract_v1.yaml
PHASE1_OUTPUT_ROOT ?= data
PHASE2_SMOKE_DIR ?= .tmp/phase2-smoke

.PHONY: check-python install lock test lint phase1-data phase1-cases phase1 phase2-smoke

check-python:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
		echo "error: PYTHON='$(PYTHON)' was not found; set PYTHON to a Python 3.11+ executable"; \
		exit 1; \
	}
	@$(PYTHON) -c 'import sys; required = (3, 11); current = sys.version_info[:2]; raise SystemExit(0 if current >= required else "error: Python 3.11+ is required, but PYTHON=$(PYTHON) resolved to %s.%s" % current)'

install: check-python
	$(PYTHON) -m pip install --requirement requirements.lock
	$(PYTHON) -m pip install --no-deps --editable .

lock: check-python
	$(PYTHON) -m piptools compile --extra dev --strip-extras \
		--output-file=requirements.lock pyproject.toml

test: check-python
	$(PYTHON) -m pytest

lint: check-python
	$(PYTHON) -m ruff check app tests notebooks

phase1-data: check-python
	$(PYTHON) -m app.data.phase1 \
		--development-config "$(DEVELOPMENT_CONFIG)" \
		--public-validation-config "$(PUBLIC_VALIDATION_CONFIG)" \
		--tool-contract "$(TOOL_CONTRACT)" \
		--output-root "$(PHASE1_OUTPUT_ROOT)" \
		--stage data

phase1-cases: check-python
	$(PYTHON) -m app.data.phase1 \
		--development-config "$(DEVELOPMENT_CONFIG)" \
		--public-validation-config "$(PUBLIC_VALIDATION_CONFIG)" \
		--tool-contract "$(TOOL_CONTRACT)" \
		--output-root "$(PHASE1_OUTPUT_ROOT)" \
		--stage cases

phase1: check-python
	$(PYTHON) -m app.data.phase1 \
		--development-config "$(DEVELOPMENT_CONFIG)" \
		--public-validation-config "$(PUBLIC_VALIDATION_CONFIG)" \
		--tool-contract "$(TOOL_CONTRACT)" \
		--output-root "$(PHASE1_OUTPUT_ROOT)" \
		--stage all

phase2-smoke: check-python
	@$(PYTHON) -m app.experiments.phase2_smoke \
		--config "$(DEVELOPMENT_CONFIG)" \
		--work-dir "$(PHASE2_SMOKE_DIR)"
