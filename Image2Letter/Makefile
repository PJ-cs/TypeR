# Makefile
SHELL = /bin/bash

# Environment
.ONESHELL:
venv:
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	python3 -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate

# Init
.ONESHELL
init: venv
	pip install -r requirements.txt && \
	dvc init && \
	dvc remote add --default myremote gdrive://1xfnkAm5QsUc9l2H4pWZbaOO4P7DbGLv1
