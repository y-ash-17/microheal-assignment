import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
INTENT_SAFETY_PATH = ROOT_DIR / "evals" / "intent_safety.yaml"
RETRIEVAL_PATH = ROOT_DIR / "evals" / "retrieval.yaml"
CACHE_PATH = ROOT_DIR / "evals" / "cache" / "model_outputs.json"

MIN_INTENT_ACCURACY = 0.75
MIN_RETRIEVAL_HIT_AT_K = 0.65
MAX_SAFETY_FALSE_NEGATIVES = 0


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "mode": "recorded_offline_cache", "by_id": {}}


async def run_evaluation(
    k: int = 3,
    *,
    replay_intent: bool = True,
    replay_retrieval: bool = False,
    save_cache: bool = False,
) -> int:
    """Run the quality gate.

    Default mode (both defaults left alone) replays the committed intent/safety
    outputs so no API key is needed, but executes retrieval live against
    PostgreSQL + pgvector using the local embedding model.  This is what makes
    the default retrieval score evidence about the current retriever rather
    than a replay of an old answer.
    """
    intent_data = _load_yaml(INTENT_SAFETY_PATH)
    retrieval_data = _load_yaml(RETRIEVAL_PATH)
    cache = _load_cache()
    cached_by_id: dict[str, Any] = dict(cache.get("by_id") or {})

    intent_cases = intent_data.get("cases") or []
    retrieval_questions = retrieval_data.get("questions") or []

    intent_mode = "REPLAY" if replay_intent else "LIVE"
    retrieval_mode = "REPLAY" if replay_retrieval else "LIVE"

    if replay_intent:
        missing_intents = [c["id"] for c in intent_cases if c["id"] not in cached_by_id]
        if missing_intents:
            print(
                "Intent replay cache is incomplete for: "
                + ", ".join(missing_intents)
                + ". Run `python -m evals.run_eval --live --save-cache`."
            )
            return 1

    classifier = None
    if not replay_intent:
        from app.classifier import SafetyAndIntentClassifier

        classifier = SafetyAndIntentClassifier(timeout_seconds=2.0)

    failing_rows: list[str] = []

    intent_correct = 0
    safety_false_negatives = 0
    tp_by_label: dict[str, int] = defaultdict(int)
    fp_by_label: dict[str, int] = defaultdict(int)
    fn_by_label: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in intent_cases:
        cid = case["id"]
        msg = case["message"]
        expected_intent = case["expected_intent"]
        expected_verdict = case["expected_safety_verdict"]

        if replay_intent:
            pred = cached_by_id[cid]
        else:
            assert classifier is not None
            cls_res = await classifier.classify(msg)
            pred = {
                "intent": cls_res["intent"],
                "safety_verdict": cls_res["verdict"],
            }
            cached_by_id[cid] = pred

        pred_intent = pred.get("intent", "UNKNOWN")
        pred_verdict = pred.get("safety_verdict", "SAFE")

        if expected_verdict == "EMERGENCY" and pred_verdict != "EMERGENCY":
            safety_false_negatives += 1
            failing_rows.append(
                f"[SAFETY_FN] {cid}: expected_verdict=EMERGENCY, got={pred_verdict} ({msg})"
            )

        confusion[expected_intent][pred_intent] += 1
        if pred_intent == expected_intent:
            intent_correct += 1
            tp_by_label[expected_intent] += 1
        else:
            fp_by_label[pred_intent] += 1
            fn_by_label[expected_intent] += 1
            failing_rows.append(
                f"[INTENT_MISMATCH] {cid}: expected={expected_intent}, got={pred_intent} ({msg})"
            )

    total_intent = max(len(intent_cases), 1)
    intent_accuracy = intent_correct / total_intent

    retrieval_hits = 0
    cached_retrieval: dict[str, Any] = dict(cache.get("retrieval_by_id") or {})
    retrieval_threshold: float | None = None

    if replay_retrieval:
        missing_retrieval = [
            q["id"] for q in retrieval_questions if q["id"] not in cached_retrieval
        ]
        if missing_retrieval:
            print(
                "Retrieval replay cache is incomplete for: "
                + ", ".join(missing_retrieval)
                + ". Run `python -m evals.run_eval --save-cache`."
            )
            return 1

        for rq in retrieval_questions:
            qid = rq["id"]
            expected_doc = rq.get("expected_doc_id")
            should_abstain = bool(rq.get("should_abstain", False))
            cached = cached_retrieval[qid]
            abstained = bool(cached.get("abstained", False))
            top_docs = list(cached.get("top_doc_ids") or [])[:k]
            if (should_abstain and abstained) or (
                not should_abstain and not abstained and expected_doc in top_docs
            ):
                retrieval_hits += 1
            else:
                failing_rows.append(
                    f"[RETRIEVAL_FAIL] {qid}: cached abstained={abstained}, docs={top_docs}"
                )
    else:
        try:
            from app.db import AsyncSessionLocal
            from app.models import Patient, Society
            from app.rag import MIN_RETRIEVAL_SCORE, retrieve_corpus_chunks

            retrieval_threshold = MIN_RETRIEVAL_SCORE
            async with AsyncSessionLocal() as session:
                for rq in retrieval_questions:
                    qid = rq["id"]
                    soc_id = rq["society_id"]
                    query = rq["query"]
                    expected_doc = rq.get("expected_doc_id")
                    should_abstain = bool(rq.get("should_abstain", False))

                    society = await session.get(Society, soc_id)
                    if not society:
                        patient = await session.get(Patient, rq.get("patient_id", ""))
                        if patient:
                            society = await session.get(Society, patient.society_id)

                    rag_out = await retrieve_corpus_chunks(
                        session,
                        query,
                        society=society,
                        society_id=soc_id,
                        top_k=k,
                        include_internal=False,
                    )
                    abstained = bool(rag_out.get("abstained"))
                    chunks = rag_out.get("chunks") or []
                    top_docs = [c.get("doc_id") for c in chunks[:k]]
                    top_scores = [round(float(c.get("score", 0.0)), 4) for c in chunks[:k]]
                    cached_retrieval[qid] = {
                        "abstained": abstained,
                        "top_doc_ids": top_docs,
                        "top_scores": top_scores,
                    }

                    if should_abstain:
                        if abstained:
                            retrieval_hits += 1
                        else:
                            failing_rows.append(
                                f"[RETRIEVAL_FAIL] {qid}: expected abstention, "
                                f"got docs={top_docs} scores={top_scores} ({query})"
                            )
                    else:
                        if not abstained and expected_doc in top_docs:
                            retrieval_hits += 1
                        else:
                            failing_rows.append(
                                f"[RETRIEVAL_FAIL] {qid}: expected={expected_doc} in top {k}, "
                                f"got docs={top_docs} scores={top_scores} ({query})"
                            )
        except Exception as exc:
            print(f"Live retrieval failed: {type(exc).__name__}: {exc}")
            print(
                "Start PostgreSQL and the seeded app (docker compose up), or run "
                "`python -m evals.run_eval --replay` for the cached retrieval gate."
            )
            return 1

    total_ret = max(len(retrieval_questions), 1)
    hit_rate_at_k = retrieval_hits / total_ret

    if save_cache:
        cache["by_id"] = cached_by_id
        cache["retrieval_by_id"] = cached_retrieval
        cache["mode"] = "recorded_offline_cache"
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    all_labels = sorted(
        set(tp_by_label.keys()) | set(fp_by_label.keys()) | set(fn_by_label.keys())
    )

    print(f"Eval mode: intent={intent_mode} retrieval={retrieval_mode}")
    if replay_intent:
        print("Intent/safety source: committed evals/cache/model_outputs.json")
    else:
        print("Intent/safety source: live SafetyAndIntentClassifier")
    if replay_retrieval:
        print("Retrieval source: committed evals/cache/model_outputs.json")
    else:
        print("Retrieval source: live pgvector + BAAI/bge-small-en-v1.5 (local model)")
        if retrieval_threshold is not None:
            print(f"Retrieval abstention threshold: {retrieval_threshold:.4f}")

    print(f"Intent accuracy: {intent_accuracy:.4f} ({intent_correct}/{len(intent_cases)})")
    for lbl in all_labels:
        tp = tp_by_label[lbl]
        fp = fp_by_label[lbl]
        fn = fn_by_label[lbl]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  class={lbl:15s} precision={prec:.4f} recall={rec:.4f}")

    print("Confusion matrix (expected -> predicted):")
    for exp_lbl in sorted(confusion.keys()):
        row_str = ", ".join(
            f"{p_lbl}:{cnt}" for p_lbl, cnt in sorted(confusion[exp_lbl].items())
        )
        print(f"  {exp_lbl:15s}: {row_str}")

    print(
        f"retrieval_hit_rate@{k}: {hit_rate_at_k:.4f} "
        f"({retrieval_hits}/{len(retrieval_questions)})"
    )
    print(f"safety_false_negatives: {safety_false_negatives}")

    if failing_rows:
        print("Failing rows:")
        for r in failing_rows:
            print(" ", r)

    if (
        safety_false_negatives > MAX_SAFETY_FALSE_NEGATIVES
        or intent_accuracy < MIN_INTENT_ACCURACY
        or hit_rate_at_k < MIN_RETRIEVAL_HIT_AT_K
    ):
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MicroHeal Care Concierge Evaluation Runner"
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval depth (default 3)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run both intent classification and retrieval live (no cache replay).",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay both intent and retrieval outputs from evals/cache (no DB/model needed).",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Save outputs from live sections into evals/cache/model_outputs.json.",
    )
    args = parser.parse_args()
    if args.live and args.replay:
        parser.error("--live and --replay are mutually exclusive")
    if args.save_cache and args.replay:
        parser.error("--save-cache has nothing to save with --replay")

    # Default is the brief's intended split: replay recorded intent/safety
    # outputs (no API key) while running retrieval live with local vectors.
    replay_intent = args.replay or not args.live
    replay_retrieval = args.replay

    return asyncio.run(
        run_evaluation(
            k=args.k,
            replay_intent=replay_intent,
            replay_retrieval=replay_retrieval,
            save_cache=args.save_cache,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
