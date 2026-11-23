import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from app.agent import run_concierge_turn
from app.booking import BookingConflictError, create_or_get_booking
from app.db import AsyncSessionLocal
from app.main import app
from app.safety import assess_safety


@pytest.mark.asyncio
async def test_duplicate_request_id_idempotency():
    req_id = "test_req_idempotency_001"
    thread_id = "test_thread_idempotency_001"

    async with AsyncSessionLocal() as s1:
        b1 = await create_or_get_booking(
            s1,
            request_id=req_id,
            thread_id=thread_id,
            patient_id="pat_003",
            clinician_id="cln_01",
            service="DOCTOR",
            subtype="GENERAL_PHYSICIAN",
            slot_start="2026-11-23T09:00:00+05:30",
        )

    async with AsyncSessionLocal() as s2:
        b2 = await create_or_get_booking(
            s2,
            request_id=req_id,
            thread_id=thread_id,
            patient_id="pat_003",
            clinician_id="cln_01",
            service="DOCTOR",
            subtype="GENERAL_PHYSICIAN",
            slot_start="2026-11-23T09:00:00+05:30",
        )

    assert b1["booking_id"] == b2["booking_id"]
    assert b2["idempotent_replay"] is True


@pytest.mark.asyncio
async def test_concurrent_double_booking_lock():
    slot = "2026-11-23T09:20:00+05:30"

    async def attempt_book(req_suffix: str):
        async with AsyncSessionLocal() as session:
            try:
                res = await create_or_get_booking(
                    session,
                    request_id=f"test_concurrent_{req_suffix}",
                    thread_id="test_thread_concurrent",
                    patient_id="pat_003",
                    clinician_id="cln_01",
                    service="DOCTOR",
                    subtype="GENERAL_PHYSICIAN",
                    slot_start=slot,
                )
                return ("ok", res["booking_id"])
            except BookingConflictError:
                return ("conflict", None)

    results = await asyncio.gather(attempt_book("a"), attempt_book("b"))
    ok_results = [r for r in results if r[0] == "ok"]
    conflict_results = [r for r in results if r[0] == "conflict"]
    assert len(ok_results) == 1
    assert len(conflict_results) == 1


@pytest.mark.asyncio
async def test_graph_restart_persistence():
    thread_id = "test_thread_restart_001"
    async with AsyncSessionLocal() as session:
        await run_concierge_turn(
            session,
            thread_id=thread_id,
            patient_id="pat_010",
            message="When does the wellness room open on Sunday?",
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/v1/threads/{thread_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == thread_id
        assert len(data["messages"]) >= 2


def test_safety_classifier_failure_defaults_to_level_1():
    def broken_classifier(_msg: str):
        raise TimeoutError("Classifier timed out")

    assessment = assess_safety("routine check question", classifier_fn=broken_classifier)
    assert assessment.level == 1
    assert assessment.call_112 is True
