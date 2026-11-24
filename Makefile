.PHONY: up down migrate seed eval eval-live eval-replay test

PYTHON ?= python3

up:
	docker compose up -d --build

down:
	docker compose down -v

# Optional: Compose already runs these on startup. Use them to re-apply
# migrations or refresh seed data without recreating the app container.
migrate:
	docker compose exec -T app alembic upgrade head

seed:
	docker compose exec -T app python -m scripts.seed_and_ingest

# Default: replay committed intent/safety outputs, but run retrieval live
# against pgvector with the local embedding model.
eval:
	docker compose exec -T app python -m evals.run_eval

eval-live:
	docker compose exec -T app python -m evals.run_eval --live

eval-replay:
	docker compose exec -T app python -m evals.run_eval --replay

test:
	docker compose exec -T app python -m pytest -v
