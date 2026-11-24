# Evaluation Report

## Verification environment

All results below were produced on **2026-11-24** from a clean database volume:
`docker compose down -v`, then `docker compose up -d --build --wait`. Compose
waited for PostgreSQL, applied both Alembic migrations (`001_initial_schema`
and `002_write_idempotency`), seeded societies/clinicians/patients/documents,
embedded the corpus with the local model, and started Uvicorn. `/healthz`
returned `{"status":"ok","db":"connected"}`.

## What each eval mode measures

The default `make eval` is deliberately split:

- **Intent/safety: replay.** It reads the committed outputs in
  `evals/cache/model_outputs.json`, so no API key is needed.
- **Retrieval: live.** It queries the seeded PostgreSQL/pgvector corpus with the
  local `BAAI/bge-small-en-v1.5` model. The reported retrieval score is therefore
  evidence about the current retriever, not a replay of an old answer.

`make eval-live` runs both sections live. `make eval-replay` replays both and is
the only mode that needs neither a database nor the embedding model.

## Results

| Mode | Intent accuracy | Live retrieval hit rate@3 | Safety false negatives |
|---|---|---|---|
| Default (`make eval`): intent replay, retrieval live | 25/25 (100%) | 9/9 (100%) | 0 |
| Fully live (`make eval-live`) | 25/25 (100%) | 9/9 (100%) | 0 |
| Fully replay (`make eval-replay`) | 25/25 (100%) | 9/9 (100%) | 0 |

The runner prints per-class precision/recall, a confusion matrix, retrieval hit
rate@k, safety false negatives, and every failing row. It exits non-zero when
intent accuracy drops below 0.75, retrieval hit rate@3 below 0.65, or any
emergency is missed.

## Retrieval threshold

The core retriever is dense-only (pgvector cosine distance; no keyword or
domain boosts). The abstention threshold is **0.63** on dense cosine similarity.
On the nine-question retrieval set, the lowest score for a correct-answer
question was **0.6699**, and the highest score for either corpus-unanswerable
question was **0.5899**; 0.63 sits near the midpoint and abstains for both
unanswerable questions without dropping a correct scoped result. The value is
configurable through `RETRIEVAL_SIMILARITY_THRESHOLD` for re-tuning.

## Required and regression tests

From the freshly seeded database, `pytest -v` reported **27 passed**. This
includes the three required scenarios:

1. Concurrent double-booking: two simultaneous requests produce exactly one
   database row and one clear conflict response.
2. Duplicate request delivery: sequential and **concurrent** deliveries of the
   same `request_id` produce one row and return that same booking to both
   callers (the concurrent case was added and now passes).
3. Restart and continue: the graph is rebuilt from nothing, prior slot-filling
   context is recovered from PostgreSQL, and the second turn books the slot.

It also covers classifier exception and timeout fail-closed behavior with zero
tool calls, non-whole-minute slot rejection, past-appointment rejection,
society/version/internal-note retrieval isolation, and malformed tool routing.

## Worst failure modes

1. **Dense-only retrieval is phrase-sensitive.** Paraphrases that are
   semantically close but lexically distant can fall below 0.63 and yield a safe
   "I don't know", even when a relevant policy exists. The abstention is safe,
   but recall is the main quality risk.
2. **Fail-closed safety can over-escalate.** A provider timeout produces
   emergency guidance and no side-effect tool, so a benign user may be told to
   call 112 during an outage. This is an intentional trade-off to avoid missing
   a real emergency.
3. **Patient identity is trusted input.** The demo does not authenticate
   `patient_id` or authorize `thread_id` ownership, which blocks real-patient
   use even though database scoping limits cross-society reads.

## First improvements with more time

1. Add authentication (JWT/OTP), derive `patient_id`/`society_id` from the
   authenticated session, and authorize thread ownership.
2. Add held-out multilingual safety data and review false-positive escalations
   with clinicians while preserving the fail-closed boundary.
3. Evaluate a hybrid lexical+dense candidate as the documented good-to-have,
   showing specific dense-only misses and keeping the current SQL scoping and
   abstention behavior unchanged.
4. Add a real pager/HIS notification adapter; until then, the API honestly
   reports `nurse_notified: false` and records only a local audit entry.
