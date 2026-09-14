# CareCopilot developer commands.
#
# Windows users without GNU make can run the underlying commands directly;
# each recipe is a single line for that reason.

VENV := backend/.venv
PY   := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
PY   := $(VENV)/Scripts/python.exe
endif

.PHONY: help venv install bootstrap migrate seed reset run test lint format check \
        web web-install web-build up down logs eval eval-judge kg \
        dataset baseline train-check train

help:
	@echo "bootstrap    Create roles, database and extensions (needs a superuser)"
	@echo "install      Create the virtualenv and install backend dependencies"
	@echo "migrate      Apply Alembic migrations"
	@echo "seed         Generate the synthetic dataset (destructive: --reset)"
	@echo "reset        migrate + seed from scratch"
	@echo "run          Start the API on http://127.0.0.1:8000"
	@echo "web          Start the frontend on http://localhost:3000"
	@echo "web-install  Install frontend dependencies"
	@echo "web-build    Production build of the frontend"
	@echo "eval         Run the 55-case evaluation set"
	@echo "eval-judge   ... and score faithfulness with an LLM judge"
	@echo "kg           Rebuild the Neo4j projection from PostgreSQL"
	@echo "test         Run the full test suite"
	@echo "lint         Ruff check"
	@echo "check        lint + test"
	@echo ""
	@echo "dataset      Build the fine-tuning instruction set and splits"
	@echo "baseline     Score the base model on the held-out split"
	@echo "train-check  Report whether this machine can train (needs a GPU)"
	@echo "train        Train the LoRA adapter"
	@echo ""
	@echo "up           Whole stack in Docker: db, api, mcp, web"
	@echo "down         Stop the stack (add ARGS=-v to drop the database)"
	@echo "logs         Follow the stack's logs"

install:
	python -m venv $(VENV) && $(PY) -m pip install -r backend/requirements.txt

# Prompts for the PostgreSQL superuser password.
bootstrap:
	psql -U postgres -f scripts/bootstrap_db.sql

migrate:
	cd backend && ../$(PY) -m alembic upgrade head

seed:
	$(PY) scripts/generate_data.py --reset

reset: migrate seed

run:
	cd backend && ../$(PY) run_server.py

test:
	cd backend && ../$(PY) -m pytest

lint:
	$(PY) -m ruff check backend scripts evaluation mcp-server

format:
	$(PY) -m ruff format backend scripts evaluation mcp-server

check: lint test

eval:
	$(PY) scripts/run_evaluation.py

# Adds a model call per cited case, so it roughly doubles the run.
eval-judge:
	$(PY) scripts/run_evaluation.py --judge

kg:
	$(PY) scripts/build_kg.py

# --- fine-tuning (PRD §22, §23) -------------------------------------------
# Behaviour only, never patient data. See docs/fine-tuning.md.

dataset:
	$(PY) scripts/build_finetuning_dataset.py

baseline:
	$(PY) scripts/analyze_base_failures.py --verbose

train-check:
	$(PY) scripts/finetune_lora.py --check

train:
	$(PY) scripts/finetune_lora.py

# --- frontend --------------------------------------------------------------

web-install:
	cd frontend && npm install

web:
	cd frontend && npm run dev

web-build:
	cd frontend && npm run build

# --- docker ----------------------------------------------------------------

up:
	docker compose up --build

down:
	docker compose down $(ARGS)

logs:
	docker compose logs -f
