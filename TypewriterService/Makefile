# Makefile
SHELL = /bin/bash
# Styling
style:
	black .
	flake8
	python3 -m isort .

# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .

