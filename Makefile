.PHONY: install test lint phase1-data phase1-cases phase1

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

lint:
	python3 -m ruff check app tests

phase1-data:
	python3 -m app.data.generator --config configs/data/synthetic_v1.yaml

phase1-cases:
	python3 -m app.data.case_generator --dataset data/synthetic/v1

phase1: phase1-data phase1-cases
