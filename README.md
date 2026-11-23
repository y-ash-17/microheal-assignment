# MicroHeal Care Concierge

A society-scoped healthcare concierge prototype built with **LangGraph**, **LangChain**, **PostgreSQL 16 + pgvector**, **SQLAlchemy 2.x**, **Alembic**, and **FastAPI** (SSE streaming + single-file HTML frontend).

## Quickstart from a Clean Clone

```bash
cp .env.example .env
docker compose up -d --build
make migrate
make seed
make eval
make test
```

Open http://localhost:8000 in your browser, or verify health:

```bash
curl http://localhost:8000/healthz
```

### Running Tests & Evaluation

```bash
# Replay committed evaluation outputs (no API key or live corpus query required)
make eval

# Run integration, concurrency, and safety tests against Postgres
make test
```

`make` commands execute inside the `app` container. A clean image build needs normal outbound access to Docker Hub and PyPI for its pinned dependencies; no model API key is needed for the offline evaluation replay.

### Optional LLM configuration

The application uses the OpenAI-compatible `ChatOpenAI` client. Leave `OPENAI_API_KEY` empty to use deterministic routing and retrieval only, or configure an OpenAI-compatible provider in `.env`:

```dotenv
OPENAI_API_KEY=your-provider-key
OPENAI_BASE_URL=https://provider.example/openai/v1
OPENAI_MODEL=your-chat-model
```

For OpenAI itself, omit `OPENAI_BASE_URL`. Keep `.env` private; it is ignored by Git. The configured model must support the chat and tool-call APIs used by the concierge.

## Architecture Overview

1. **Safety-First LangGraph (`app/graph.py`)**:
   - `safety_node` runs before routing or tool execution.
   - **EMERGENCY** (Level 1: 5-min SLA, Level 2: 30-min SLA, Level 3: 240-min SLA): Automatically persists to `escalations` AND dispatches an alert to the on-duty wellness room nurse (`app/notifications.py`).
   - Supports English, Hinglish (*"saans nahi aa rahi hai"*, *"chhati me dard"*, *"tez bleeeding"*), and Devanagari Hindi (*"सांस नहीं आ रही"*, *"सीने में दर्द"*).
   - **MEDICAL_ADVICE**: Refuses diagnosis/prescriptions/dosages and guides the patient to a doctor consultation.
   - **SAFE**: Routes through `router_node` to LLM/tools for booking, symptom logging, service listing, and clinic policies.

2. **Database-Enforced Invariants (`alembic/versions/`)**:
   - Partial unique index `uq_service_requests_active_clinician_slot` on `(clinician_id, scheduled_for)` WHERE `status IN ('CONFIRMED', 'PENDING')` prevents double-booking under concurrency.
   - `UNIQUE` constraint on `service_requests.request_id`, `symptom_logs.request_id`, and `escalations.request_id` ensures idempotent retries.
   - `CHECK (severity >= 1 AND severity <= 5)` enforces symptom severity bounds.

3. **Clinician & Booking Validation (`app/booking.py` & `app/tools.py`)**:
   - **Patient Ownership Check**: Reusing a `request_id` across different patient IDs is rejected (`ValueError`).
   - **Service & Specialty Matching**: Validates that clinician service (DOCTOR vs LAB) and specialty/subtypes match the request.
   - **Timezone Gap Elimination**: Canonicalizes all datetime strings to standard ISO clinic time (`Asia/Kolkata` `+05:30`), so identical instants in UTC (`04:30Z`) and IST (`10:00+05:30`) map to the exact same database slot key.

4. **Symptom Logging Chat Integration (`app/graph.py`)**:
   - Patients can record symptoms and severities directly via natural language (e.g. *"Log symptom: mild fever severity 2"* or Hinglish *"bukhar hai severity 3"*).
   - Automatically executes `log_symptom` tool and logs to `symptom_logs` in PostgreSQL.

5. **Dense Neural Embeddings & Scoping (`app/rag.py` & `app/scoping.py`)**:
   - Uses a dense neural transformer embedding model (`BAAI/bge-small-en-v1.5` via `fastembed`) when available, with deterministic keyword scoring and abstention as a safe fallback.
   - **Internal Notes Isolation**: Documents marked `internal` (such as `staff-notes-internal.md`) are strictly excluded from patient search results.
   - **Society & Version Isolation**: Every query requires `society_id` and filters active (non-superseded) documents applicable to that society.

6. **Evaluation Suite (`evals/run_eval.py`)**:
   - `make eval` replays committed classifier and retrieval outputs, so the quality gate runs without an API key or a live corpus query.
   - `docker compose exec -T app python -m evals.run_eval --live --save-cache` exercises the active classifier and pgvector corpus, then refreshes that committed replay artifact.

## Retry After a Crash

If a booking commits but the process dies before its SSE reply reaches the patient, the patient can resend the same `request_id`. The idempotency lookup returns the existing booking with `idempotent_replay: true`; it creates no second row and does not consume the clinician slot twice. That is the intended outcome: the patient sees the committed result consistently after an uncertain network or process failure.

## Deliberate Cuts

The Core assignment is complete. I intentionally did not implement the optional approval interrupt/resume flow, hybrid retrieval, latency JSONL metrics, or extra HTTP resources beyond the fixed contract. Those would add breadth but not improve the required booking, safety, retrieval-isolation, and restart guarantees within the timebox. With no LLM configured, RAG fallback returns the scoped source text; an LLM-backed response is instructed to answer in the user's language.

## Verified local checks

On 2026-11-23, the Compose image built successfully, `/healthz` reported a connected database, `make eval` passed (25/25 intents, 9/9 retrieval checks, zero safety false negatives), and `make test` passed (24 tests).
