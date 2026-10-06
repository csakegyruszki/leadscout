.PHONY: install test lint run eval

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	ruff check .

run:
	python -m leadscout.cli batch samples/leads.json

eval:
	python -m leadscout.cli eval evals/compliance_cases.json
