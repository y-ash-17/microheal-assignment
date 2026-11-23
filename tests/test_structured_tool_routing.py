import uuid

from app.graph import _route_bound_tool_call


def _state(request_id: str) -> dict:
    return {
        "patient_id": "pat_001",
        "society_id": "soc_shivalik",
        "request_id": request_id,
    }


def test_bound_tool_call_routes_booking_with_trusted_state_ids():
    request_id = f"req-{uuid.uuid4()}"
    route = _route_bound_tool_call(
        {
            "name": "book_service",
            "args": {
                "patient_id": "attacker-selected-id",
                "request_id": "attacker-selected-request",
                "service": "DOCTOR",
                "subtype": "CARDIOLOGY",
                "scheduled_for": "2027-03-02T10:00:00+05:30",
            },
            "id": "call_booking",
        },
        state=_state(request_id),
        user_message="Book cardiology for 2027-03-02 10:00",
        pending=None,
    )

    assert route["intent"] == "BOOK_SERVICE"
    assert route["pending_booking"] == {
        "service": "DOCTOR",
        "subtype": "CARDIOLOGY",
        "scheduled_for": "2027-03-02T10:00:00+05:30",
    }


def test_malformed_bound_tool_call_never_becomes_a_write_action():
    route = _route_bound_tool_call(
        {"name": "book_service", "args": {"service": "DOCTOR"}, "id": "bad"},
        state=_state(f"req-{uuid.uuid4()}"),
        user_message="Book cardiology for 2027-03-02 10:00",
        pending=None,
    )

    assert "error" in route
    assert "pending_booking" not in route
