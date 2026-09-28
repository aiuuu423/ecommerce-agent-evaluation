PYTHON ?= python3

.PHONY: check-python install lock test lint phase1-data phase1-cases phase1

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
	$(PYTHON) -m ruff check app tests

phase1-data: check-python
	$(PYTHON) -m app.data.generator --config configs/data/synthetic_v1.yaml
	$(PYTHON) -m app.data.generator --config configs/data/synthetic_holdout_v1.yaml

phase1-cases: check-python
	$(PYTHON) -m app.data.case_generator \
		--development-dataset data/synthetic/v1 \
		--holdout-dataset data/synthetic/holdout-v1 \
		--tool-contract configs/evaluation/tool_contract_v1.yaml \
		--output data/evaluation_cases/v1

phase1: check-python phase1-data phase1-cases
