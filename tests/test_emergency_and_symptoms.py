import uuid
import pytest
from app.agent import run_concierge_turn
from app.notifications import NOTIFICATION_DISPATCH_LOG
from app.scoping import filter_scoped_chunks


@pytest.mark.asyncio
async def test_emergency_dispatches_nurse_notification(toolbox):
    """Emergency escalation triggers the duty nurse notification workflow."""
    start_count = len(NOTIFICATION_DISPATCH_LOG)
    req_id = f"req-emg-{uuid.uuid4()}"

    res = await toolbox.escalate_emergency(
        patient_id="pat_001",
        transcript="Severe chest pain and difficulty breathing",
        request_id=req_id,
        level=1,
        sla_minutes=5,
    )

    assert res.get("status") == "OPEN"
    assert res.get("nurse_notified") is True
    assert res.get("notification_channel") == "WELLNESS_ROOM_DESK"
    assert len(NOTIFICATION_DISPATCH_LOG) == start_count + 1

    last_alert = NOTIFICATION_DISPATCH_LOG[-1]
    assert last_alert["patient_id"] == "pat_001"
    assert last_alert["level"] == 1
    assert last_alert["sla_minutes"] == 5
    assert last_alert["channel"] == "WELLNESS_ROOM_DESK"


@pytest.mark.asyncio
async def test_internal_staff_notes_excluded_from_patient_retrieval(toolbox):
    """Internal staff notes document is strictly excluded from patient search results."""
    results = await toolbox.search_clinic_docs("wellness room staff notes collection rounds", society_id="soc_lotus", k=10)
    for doc in results:
        assert doc["doc_id"] != "staff-notes-internal"
        assert "staff-notes" not in doc["doc_id"].lower()
        assert "internal" not in doc["doc_id"].lower()


@pytest.mark.asyncio
async def test_symptom_logging_chat_flow(db_session_factory, pg_checkpointer):
    """Symptom logging is connected to the concierge chat graph."""
    thread_id = f"thread-sym-{uuid.uuid4().hex[:8]}"
    turn_res = await run_concierge_turn(
        db_session_factory,
        thread_id=thread_id,
        patient_id="pat_001",
        message="Please log my symptom: mild fever with severity 2",
        checkpointer=pg_checkpointer,
    )

    tool_calls = turn_res.get("tool_calls") or []
    sym_calls = [c for c in tool_calls if c.get("tool") == "log_symptom"]
    assert len(sym_calls) >= 1
    assert sym_calls[0]["result"]["severity"] == 2
    assert "Recorded in your care log" in turn_res.get("reply", "")
