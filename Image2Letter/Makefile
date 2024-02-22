.PHONY: venv-dev venv ensure-venv lint test dvc-push
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

mypy: ensure-venv
	$(PYTHON) -m mypy . --ignore-missing-imports --follow-imports=skip --strict-optional

lint: ensure-venv
	ruff format .
	ruff check .

test: ensure-venv  ## Run tests
	$(PYTHON) -m pytest -v --html=.data/testFiles/report.html --css=.data/testFiles/assets/style.css
	firefox .data/testFiles/report.html

dvc-push: 
	dvc add .data
	dvc push