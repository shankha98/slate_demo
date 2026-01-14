.PHONY: install run clean build docker-run test format lint fix

install:
	clear
	uv sync

run:
	clear
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	clear
	uv run pytest tests/

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ty check app tests

fix:
	uv run ruff check --fix .
	uv run ruff format .

clean:
	rm -rf .venv
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

build:
	clear
	docker build -t slate-demo .

docker-run:
	clear
	docker run -p 8000:8000 slate-demo

cloud_run_deploy:
	clear
	gcloud beta builds submit --config cloudbuild.yaml .
