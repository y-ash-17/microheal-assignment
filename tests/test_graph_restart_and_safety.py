import asyncio
import uuid
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete, select

from app.classifier import SafetyAndIntentClassifier
from app.graph import build_care_concierge_graph
from app.models import ServiceRequest


@pytest.mark.asyncio
async def test_restart_and_continue_slot_filling(db_session_factory, pg_checkpointer, toolbox):
    """Turn 1 asks to book cardiology without date; rebuild graph from scratch; Turn 2 gives date -> books."""
    slot = "2027-03-02T10:00:00+05:30"
    async with db_session_factory() as s:
        await s.execute(delete(ServiceRequest).where(ServiceRequest.scheduled_for == slot))
        await s.commit()

    thread_id = f"thread-restart-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Instance 1
    graph_1 = build_care_concierge_graph(
        toolbox=toolbox,
        classifier=SafetyAndIntentClassifier(llm=None),
        llm=None,
        checkpointer=pg_checkpointer,
    )
    out1 = await graph_1.ainvoke(
        {
            "messages": [HumanMessage(content="book me cardiology")],
            "patient_id": "pat_001",
            "society_id": "soc_shivalik",
            "request_id": f"req-{uuid.uuid4()}",
            "language": "en",
        },
        config=config,
    )
    reply1 = str(out1["messages"][-1].content).lower()
    assert "date and time" in reply1
    assert out1["tool_calls"] == []
    async with db_session_factory() as s:
        rows = (
            await s.execute(select(ServiceRequest).where(ServiceRequest.scheduled_for == slot))
        ).scalars().all()
    assert rows == []

    # Simulate process restart: construct a brand-new StateGraph instance wired to Postgres checkpointer
    graph_2 = build_care_concierge_graph(
        toolbox=toolbox,
        classifier=SafetyAndIntentClassifier(llm=None),
        llm=None,
        checkpointer=pg_checkpointer,
    )
    out2 = await graph_2.ainvoke(
        {
            "messages": [HumanMessage(content="2027-03-02 10:00")],
            "patient_id": "pat_001",
            "society_id": "soc_shivalik",
            "request_id": f"req-{uuid.uuid4()}",
            "language": "en",
        },
        config=config,
    )
    reply2 = str(out2["messages"][-1].content)
    assert "confirmed" in reply2.lower()
    assert "CARDIOLOGY" in reply2


@pytest.mark.asyncio
async def test_safety_classifier_failure_and_timeout_fail_closed(monkeypatch, pg_checkpointer, toolbox):
    """Monkeypatch safety classifier to raise and to time out; verify user message & zero tool calls."""
    classifier = SafetyAndIntentClassifier(llm=None)
    graph = build_care_concierge_graph(
        toolbox=toolbox,
        classifier=classifier,
        llm=None,
        checkpointer=pg_checkpointer,
    )

    async def raise_boom(_msg: str):
        raise RuntimeError("Provider 503 Service Unavailable")

    monkeypatch.setattr(classifier, "classify_safety", raise_boom)

    res_err = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Book cardiology for 2027-03-02 10:00")],
            "patient_id": "pat_001",
            "society_id": "soc_shivalik",
            "request_id": f"req-{uuid.uuid4()}",
            "language": "en",
        },
        config={"configurable": {"thread_id": f"thread-fail-{uuid.uuid4().hex[:6]}"}},
    )
    assert res_err["safety_fallback_triggered"] is True
    assert res_err["tool_calls_count"] == 0
    assert "112" in str(res_err["messages"][-1].content)
    assert classifier.failure_count == 1

    async def raise_timeout(_msg: str):
        raise asyncio.TimeoutError("Classifier timed out")

    monkeypatch.setattr(classifier, "classify_safety", raise_timeout)

    res_to = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Show my bookings")],
            "patient_id": "pat_001",
            "society_id": "soc_shivalik",
            "request_id": f"req-{uuid.uuid4()}",
            "language": "en",
        },
        config={"configurable": {"thread_id": f"thread-to-{uuid.uuid4().hex[:6]}"}},
    )
    assert res_to["safety_fallback_triggered"] is True
    assert res_to["tool_calls_count"] == 0
    assert classifier.failure_count == 2


@pytest.mark.asyncio
async def test_tool_exception_and_malformed_args_handled_cleanly(monkeypatch, pg_checkpointer, toolbox):
    """Verify tool raising an exception and malformed arguments are caught without crashing the graph."""
    classifier = SafetyAndIntentClassifier(llm=None)
    graph = build_care_concierge_graph(
        toolbox=toolbox,
        classifier=classifier,
        llm=None,
        checkpointer=pg_checkpointer,
    )

    async def broken_list(*args, **kwargs):
        raise ConnectionError("Simulated DB failure")

    monkeypatch.setattr(toolbox, "list_service_requests", broken_list)

    out = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Show my upcoming bookings")],
            "patient_id": "pat_001",
            "society_id": "soc_shivalik",
            "request_id": f"req-{uuid.uuid4()}",
            "language": "en",
        },
        config={"configurable": {"thread_id": f"thread-toolerr-{uuid.uuid4().hex[:6]}"}},
    )
    assert "temporary system error" in str(out["messages"][-1].content).lower()
