.PHONY: install test lint clean

install:
	pip install -e .[dev]

test:
	pytest tests/

lint:
	ruff check src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
