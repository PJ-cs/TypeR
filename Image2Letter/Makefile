.PHONY: venv-dev venv ensure-venv isort black mypy flake8 lint test dvc-push
.ONESHELL:

# Load environment variables from .env file if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

VENV=.venv
PYTHON=$(VENV)/bin/python3

ensure-venv:
	@test -d $(VENV) || (echo "Virtual environment not found. Please run 'make venv-dev' or 'make venv' first." && exit 1)

cmd-exists-%:
	@hash $(*) > /dev/null 2>&1 || \
		(echo "ERROR: '$(*)' must be installed and available on your PATH."; exit 1)

venv-dev: requirements.txt requirements-dev.txt  ## Create virtual dev environment
	@make cmd-exists-python
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

venv: requirements.txt  ## Create virtual environment
	@make cmd-exists-python
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

isort: ensure-venv
	$(PYTHON) -m isort .

black: ensure-venv
	$(PYTHON) -m black --line-length=120 .

mypy: ensure-venv
	$(PYTHON) -m mypy . --ignore-missing-imports --follow-imports=skip --strict-optional

flake8: ensure-venv
	$(PYTHON) -m flake8 .

lint: isort black mypy flake8

test: ensure-venv  ## Run tests
	$(PYTHON) -m pytest

dvc-push: 
	dvc add data
	dvc push