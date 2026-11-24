# MicroHeal Care Concierge

A society-scoped healthcare concierge prototype built with **LangGraph**, **LangChain**, **PostgreSQL 16 + pgvector**, **SQLAlchemy 2.x**, **Alembic**, and **FastAPI** (SSE streaming + single-file HTML frontend).

## Quickstart from a Clean Clone

```bash
cp .env.example .env
docker compose up -d --build
make eval
make test
```

`docker compose up` starts PostgreSQL and the app, waits for the database,
applies Alembic migrations, seeds the demo data, embeds the corpus, and then
serves FastAPI. Migrations and seeding are therefore not separate prerequisites.
`make migrate` and `make seed` remain available for re-running those steps on an
existing container.

Open http://localhost:8000 in your browser, or verify health:

```bash
curl http://localhost:8000/healthz
```

### Running Tests & Evaluation

```bash
# Default: replay committed intent/safety outputs (no API key) and run
# retrieval LIVE against pgvector with the local bge-small model.
make eval

# Fully live: also runs the active classifier instead of replaying its outputs.
make eval-live

# Fully offline: replays both intent and retrieval outputs; no database/model.
make eval-replay

# Run integration, concurrency, and safety tests against Postgres
make test
```

`make` commands execute inside the `app` container. A clean image build needs
normal outbound access to Docker Hub and PyPI for its pinned dependencies; the
Dockerfile also warms the local embedding model. No model API key is needed for
the default evaluation because intent outputs are replayed.

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
   - **EMERGENCY** (Level 1: 5-min SLA, Level 2: 30-min SLA, Level 3: 240-min SLA): Automatically persists to `escalations` and appends a **local audit record** (`app/notifications.py`). No real nurse pager/SMS/WhatsApp integration is configured, so the API returns `nurse_notified: false` and the reply says the escalation was stored rather than claiming a nurse was contacted.
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
   - **Minute-Granular Slots**: A slot with non-zero seconds or microseconds is rejected instead of being silently rounded to a different instant.
   - **No Past Bookings**: A new booking must be for a future instant; idempotent replays of an existing booking remain valid.

4. **Symptom Logging Chat Integration (`app/graph.py`)**:
   - Patients can record symptoms and severities directly via natural language (e.g. *"Log symptom: mild fever severity 2"* or Hinglish *"bukhar hai severity 3"*).
   - Automatically executes `log_symptom` tool and logs to `symptom_logs` in PostgreSQL.

5. **Dense-Only Retrieval & Scoping (`app/rag.py` & `app/scoping.py`)**:
   - Core retrieval is **dense-only**: pgvector orders chunks by cosine distance to the query embedding from the local `BAAI/bge-small-en-v1.5` model. Keyword scores, domain-term boosts, and annex/coverage boosts were removed from the core path.
   - **Internal Notes Isolation**: Documents marked `internal` (such as `staff-notes-internal.md`) are strictly excluded from patient search results.
   - **Society & Version Isolation**: Every query requires `society_id` and filters active (non-superseded) documents applicable to that society in the SQL `WHERE` clause before ranking.
   - **Abstention**: Chunks below the dense cosine threshold are discarded, and an empty result returns the "I don't know" message with a nurse offer instead of answering from model memory.

6. **Evaluation Suite (`evals/run_eval.py`)**:
   - The default `make eval` replays committed intent/safety outputs but executes retrieval **live** against pgvector and the local embedding model, so the reported retrieval score measures the current retriever.
   - `make eval-live` runs the active classifier and retrieval live; `make eval-replay` replays both from `evals/cache/`.
   - `--save-cache` refreshes the committed replay artifact after a live run.

## Retry After a Crash

If a booking commits but the process dies before its SSE reply reaches the patient, the patient can resend the same `request_id`. The idempotency lookup returns the existing booking with `idempotent_replay: true`; it creates no second row and does not consume the clinician slot twice. That is the intended outcome: the patient sees the committed result consistently after an uncertain network or process failure.

## Known Limitations (Demo Scope)

- **Patient access is not authenticated.** `POST /v1/chat` trusts caller-supplied
  `patient_id`, and `GET /v1/threads/{thread_id}` trusts a caller-supplied
  thread ID. There is no login, token, or ownership check. A caller who knows or
  guesses an ID can act as or read that patient's thread. This is acceptable for
  a controlled assignment demo but blocks real-patient use; production needs
  authentication (for example JWT or OTP), `patient_id` derived from the
  authenticated session, and an authorization check on thread ownership.
- **No real nurse notification is sent.** An emergency writes the required
  `escalations` row and appends a process-local audit record. No pager, SMS,
  WhatsApp, or hospital-HIS integration exists, so every response reports
  `nurse_notified: false` and the chat reply says the escalation was stored
  rather than claiming a nurse was contacted.
- **Retrieval is scoped per society, not per authenticated patient.** Society
  and document-version filters are enforced in SQL, but the patient's society is
  looked up from the supplied patient ID.

## Deliberate Cuts

The Core assignment is complete. I intentionally did not implement the optional
approval interrupt/resume flow, hybrid lexical+dense retrieval, latency JSONL
metrics, external nurse paging/authentication, or extra HTTP resources beyond
the fixed contract. Those would add breadth but not improve the required
booking, safety, retrieval-isolation, and restart guarantees within the
timebox. The core retriever is dense-only as specified; a hybrid ranker remains
a documented good-to-have. With no LLM configured, RAG returns the scoped source
text; an LLM-backed response is instructed to answer in the user's language.

## Verified local checks

Verified on 2026-11-24 from a rebuilt image and a freshly created PostgreSQL
volume: `docker compose up` applied migrations and seeded automatically;
`/healthz` reported a connected database; the default `make eval` replayed
intent/safety outputs and ran retrieval live against pgvector; and the required
concurrency, duplicate-delivery, restart, and safety tests passed. Exact output
is recorded in `EVAL_REPORT.md`.
