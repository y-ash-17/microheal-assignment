# Architectural Decisions

## 1. Safety-classifier failure: fail closed

- **Choice:** Fail closed. A provider error or timeout is logged as `safety_classifier_failure` and stops before every side-effect tool.
- **Patient outcome:** The patient receives emergency guidance to call 112; the system does not create an escalation row for an unconfirmed classifier result.
- **Rejected option:** Fail open and continue to routing, retrieval, or a write tool after an unavailable safety classifier.
- **Why reject it:** An emergency or medication/dosage request could bypass the only safety boundary during an outage.
- **Cost:** Benign users can be incorrectly blocked while the provider is unavailable.
- **Mitigation:** The failure counter and structured log make this visible and measurable; actual detected emergencies still persist through `escalate_emergency`.
- **Proof:** Tests monkeypatch both an exception and a timeout, assert 112 guidance, and assert zero tool calls.

## 2. Double-booking: enforce the invariant in PostgreSQL

- **Choice:** A partial unique index on `(clinician_id, scheduled_for)` for `CONFIRMED` and `PENDING` requests is the final concurrency boundary.
- **Supporting choice:** Canonicalise every slot to Asia/Kolkata ISO time before querying, locking, or inserting.
- **Rejected option:** Python-only locks or a check-then-insert availability query.
- **Why reject it:** They do not protect multiple processes, replicas, retries, restarts, or another database writer.
- **Cost:** The application must catch `IntegrityError` and turn it into a patient-readable conflict response.
- **Outcome:** One concurrent request commits; the other receives a clear `SLOT_TAKEN` message. Equivalent UTC and IST inputs collide on the same slot.
- **Proof:** Concurrent booking, timezone-normalisation, and repeated-request tests run against PostgreSQL.

## 3. LLM providers: use an OpenAI-compatible boundary

- **Choice:** The application creates its chat model through `ChatOpenAI` and accepts an optional `OPENAI_BASE_URL`.
- **Outcome:** It supports OpenAI directly and compatible hosted providers without provider-specific application code. `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` are deployment environment variables; secrets are not committed.
- **Fallback:** If no key is configured, or a model call fails, deterministic intent routing and scoped retrieval still provide a safe response. Safety classification remains fail-closed.
- **Cost:** Tool-calling and response quality vary by provider/model, so a deployment must use a model that supports the required OpenAI-compatible chat and tool-call APIs.

## 4. Retrieval startup behavior: preserve correctness over availability of a remote model download

- **Choice:** `fastembed` provides dense embeddings when its model is available; retrieval also has a deterministic keyword-scoring component and abstains below the configured threshold.
- **Outcome:** The service can still return scoped policy results when the embedding model is unavailable, rather than exposing cross-society data or inventing an answer.
- **Deployment note:** The local Compose configuration persists the Hugging Face cache in `modelcache` to avoid repeated model downloads after container restarts.
