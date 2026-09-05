.PHONY: setup data features splits experiments reports test api docker-build docker-up all clean

PYTHONPATH := src
export PYTHONPATH

setup:
	pip install --break-system-packages -e ".[dev]"

data:
	python -m abusering.data.generate --profile demo --seed 42 --out data_store

data-full:
	python -m abusering.data.generate --profile full --seed 42 --out data_store

features:
	python -m abusering.features.build_cli --data data_store --out data_store

splits:
	python -m abusering.splits.ring_split --data data_store --out data_store --seed 42

experiments:
	python -m abusering.evaluation.run_experiments --data data_store --artifacts artifacts \
		--reports reports --cost-config configs/cost_config.json --optuna-trials 20 --seed 42

reports:
	python -m abusering.reports.generate --artifacts artifacts --data data_store --out reports

evaluate: experiments reports

test:
	pytest tests/ -v

api:
	uvicorn abusering.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && python3 -m http.server 8080

all: data features splits experiments reports test

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf data_store artifacts/*.pkl artifacts/*.db artifacts/mlflow.db artifacts/experiments.json reports/*.md
