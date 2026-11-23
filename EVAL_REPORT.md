# Evaluation Report

## Verified offline run

On 2026-11-23, `docker compose exec -T app python -m evals.run_eval` replayed the committed outputs in `evals/cache/model_outputs.json`, without an API key. The run reported 25/25 intent labels correct (100.00% intent accuracy), retrieval hit rate@3 of 9/9 (100.00%), and 0 safety false negatives. It prints class-level precision/recall and a confusion matrix; it exits non-zero if intent accuracy drops below 0.75, retrieval hit rate@3 below 0.65, or any emergency is missed.

The retrieval abstention threshold is 0.35. It was selected on the retrieval evaluation set because it retained every expected scoped policy result while abstaining for both corpus-unanswerable questions; lower scores admitted off-topic chunks.

## Worst failure modes

1. A rules-plus-provider safety classifier can over-escalate when its provider times out. This prototype intentionally fails closed, so a benign user may be told to call emergency services during an outage. That has a usability cost, but prevents an emergency or dosage request from silently bypassing the gate.
2. Dense retrieval quality depends on corpus wording and embedding quality. An unfamiliar phrasing can fall below the threshold and produce a safe "I don't know" even when an indirectly relevant policy exists.

## First improvement with more time

I would add held-out multilingual safety data and review false-positive escalations with clinicians, then tune the classifier while preserving the fail-closed boundary. For retrieval, I would add a separately evaluated hybrid lexical+dense ranker and expand the corpus aliases, while keeping the SQL society/version predicates and abstention behavior unchanged.

## Final local verification

On 2026-11-23, the current Compose stack was started successfully and the full database-backed suite passed: **24 passed** in 7.62 seconds. This includes concurrent double booking, same-chat `request_id` replay, restart-and-continue, safety-provider exception/timeout, scoped retrieval, version filtering, and internal-note exclusion.

The Compose configuration validates and the application image builds locally. `/healthz` returned `{"status":"ok","db":"connected"}` after startup. A direct, non-personal OpenAI-compatible model smoke test also succeeded using the deployment environment configuration.

The cached evaluation is a reproducible quality gate, not a claim that every provider/model will return identical online output. To exercise the active corpus and configured model, run `docker compose exec -T app python -m evals.run_eval --live` in an environment approved to send the evaluation prompts to that provider.
