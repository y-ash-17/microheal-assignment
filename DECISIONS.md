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

## 4. Retrieval ranker: dense cosine only

- **Choice:** The core retriever ranks scoped `doc_chunks` by pgvector cosine distance to a `BAAI/bge-small-en-v1.5` query embedding. No keyword score, domain-term boost, annex boost, or coverage boost is added.
- **Rejected option:** Blending dense similarity with keyword overlap and hand-written domain/annex boosts in the core path.
- **Why reject it:** The brief specifies dense search for the core. The blended score was not a similarity score, so its threshold could not be interpreted as evidence about embedding quality, and it masked cases where dense retrieval alone failed. A hybrid lexical+dense ranker remains the documented good-to-have and would need its own ablation.
- **Threshold:** `RETRIEVAL_SIMILARITY_THRESHOLD=0.63` on dense cosine similarity. On the nine-question retrieval set the lowest correct-answer score was 0.6699 and the highest unanswerable-question score was 0.5899, so 0.63 sits near the midpoint and abstains for both corpus-unanswerable questions; it is configurable for re-tuning.
- **Cost:** Dense-only retrieval is sensitive to wording and needs the local embedding model. If the model is unavailable the retriever abstains instead of silently falling back to keyword matching.
- **Isolation:** Society applicability, `superseded_by IS NULL`, and internal-note exclusion are SQL `WHERE` predicates applied before ranking, not Python post-filters.
- **Proof:** The default `make eval` now runs retrieval live against pgvector; `make eval-replay` exists only for a deliberately offline replay.

## 5. Patient identity: trusted demo input, not authentication

- **Choice:** The demo accepts `patient_id` in the chat request and `thread_id` in the thread lookup, and uses them as supplied. Society scoping and booking ownership still constrain what the tools read and write.
- **Rejected option:** Implementing full resident authentication (JWT/OTP, session-derived patient identity, thread ownership checks) inside the assignment timebox.
- **Why reject it for now:** The fixed contract has no auth fields or login flow, and inventing one would expand the surface beyond the graded core. The assignment is a controlled demo, so the risk can be stated rather than hidden.
- **Cost:** This blocks real-patient use. A caller who knows another patient's ID or thread ID could act as or read that patient's conversation; there is no proof that the caller owns the identity.
- **Production fix:** Derive `patient_id` and `society_id` from an authenticated session, authorize every `thread_id` against that identity, and keep the existing database-level scoping as defence in depth.
