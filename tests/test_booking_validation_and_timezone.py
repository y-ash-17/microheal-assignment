import uuid
import pytest
from sqlalchemy import delete

from app.booking import BookingConflictError, create_or_get_booking
from app.models import ServiceRequest


@pytest.mark.asyncio
async def test_reused_request_id_cross_patient_rejected(db_session_factory, toolbox):
    """A request_id already used by pat_001 cannot be replayed by pat_002."""
    slot = "2027-02-24T10:00:00+05:30"
    shared_req_id = f"req-shared-{uuid.uuid4()}"

    async with db_session_factory() as s:
        await s.execute(delete(ServiceRequest).where(ServiceRequest.scheduled_for == slot))
        await s.commit()

    # First booking by pat_001 succeeds
    b1 = await toolbox.book_service("pat_001", "DOCTOR", "GENERAL_PHYSICIAN", slot, shared_req_id)
    assert b1.get("status") == "CONFIRMED"

    # Attempt to reuse same request_id for pat_002 must be rejected
    async with db_session_factory() as session:
        with pytest.raises(ValueError, match="belongs to another patient"):
            await create_or_get_booking(
                session,
                request_id=shared_req_id,
                patient_id="pat_002",
                clinician_id="cln_01",
                service="DOCTOR",
                subtype="GENERAL_PHYSICIAN",
                scheduled_for=slot,
            )


@pytest.mark.asyncio
async def test_clinician_service_and_specialty_validation(db_session_factory):
    """Booking validates clinician service and specialty."""
    async with db_session_factory() as session:
        # 1. Service mismatch: booking LAB service for a DOCTOR clinician (cln_01)
        with pytest.raises(ValueError, match="provides DOCTOR, not requested service LAB"):
            await create_or_get_booking(
                session,
                request_id=f"req-{uuid.uuid4()}",
                patient_id="pat_001",
                clinician_id="cln_01",
                service="LAB",
                subtype="CBC",
                scheduled_for="2027-02-17T10:00:00+05:30",
            )

        # 2. Specialty mismatch: booking CARDIOLOGY for general physician (cln_01)
        with pytest.raises(ValueError, match="does not offer specialty/test 'CARDIOLOGY'"):
            await create_or_get_booking(
                session,
                request_id=f"req-{uuid.uuid4()}",
                patient_id="pat_001",
                clinician_id="cln_01",
                service="DOCTOR",
                subtype="CARDIOLOGY",
                scheduled_for="2027-02-17T10:00:00+05:30",
            )


@pytest.mark.asyncio
async def test_timezone_gap_closed_between_utc_and_ist(db_session_factory, toolbox):
    """Booking same instant in UTC (04:30Z) and IST (10:00+05:30) is detected as a conflict."""
    slot_ist = "2027-02-17T10:00:00+05:30"
    slot_utc = "2027-02-17T04:30:00Z"  # Exact same instant in UTC

    async with db_session_factory() as s:
        await s.execute(delete(ServiceRequest).where(ServiceRequest.scheduled_for == slot_ist))
        await s.commit()

    # Book via IST string
    b1 = await toolbox.book_service("pat_001", "DOCTOR", "GENERAL_PHYSICIAN", slot_ist, f"req-{uuid.uuid4()}")
    assert b1.get("status") == "CONFIRMED"

    # Attempt to book exact same slot using UTC string
    b2 = await toolbox.book_service("pat_003", "DOCTOR", "GENERAL_PHYSICIAN", slot_utc, f"req-{uuid.uuid4()}")
    assert b2.get("status") == "SLOT_TAKEN"
    assert "already booked" in b2.get("error", "").lower() or "already taken" in b2.get("error", "").lower()


@pytest.mark.asyncio
async def test_slot_with_seconds_is_rejected_not_rounded(db_session_factory):
    """A slot with non-zero seconds is rejected instead of silently truncated."""
    async with db_session_factory() as session:
        with pytest.raises(ValueError, match="whole minute"):
            await create_or_get_booking(
                session,
                request_id=f"req-{uuid.uuid4()}",
                patient_id="pat_001",
                clinician_id="cln_01",
                service="DOCTOR",
                subtype="GENERAL_PHYSICIAN",
                scheduled_for="2027-02-15T09:00:30+05:30",
            )


@pytest.mark.asyncio
async def test_past_appointment_is_rejected(db_session_factory):
    """A booking in the past is rejected before any row is written."""
    async with db_session_factory() as session:
        with pytest.raises(ValueError, match="past"):
            await create_or_get_booking(
                session,
                request_id=f"req-{uuid.uuid4()}",
                patient_id="pat_001",
                clinician_id="cln_01",
                service="DOCTOR",
                subtype="GENERAL_PHYSICIAN",
                scheduled_for="2020-01-06T09:00:00+05:30",
            )
