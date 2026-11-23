.PHONY: up down migrate seed eval test

PYTHON ?= python3

up:
	docker compose up -d --build

down:
	docker compose down -v

migrate:
	docker compose exec -T app alembic upgrade head

seed:
	docker compose exec -T app python -m scripts.seed_and_ingest

eval:
	docker compose exec -T app python -m evals.run_eval

test:
	docker compose exec -T app python -m pytest -v
