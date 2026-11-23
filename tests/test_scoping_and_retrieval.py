import uuid
import pytest
from app.scoping import list_scoped_service_requests


@pytest.mark.asyncio
async def test_cross_society_isolation_for_bookings_and_docs(db_session_factory, toolbox):
    """Patient of soc_shivalik cannot read soc_lotus bookings or documents."""
    # Book a Lotus Tower slot for pat_009 (soc_lotus)
    lotus_slot = "2027-01-13T10:00:00+05:30"  # Wednesday Dr. Prakash Nair at Lotus Tower
    await toolbox.book_service("pat_009", "DOCTOR", "GENERAL_PHYSICIAN", lotus_slot, f"req-{uuid.uuid4()}")

    async with db_session_factory() as s:
        # Attempt to read pat_009's bookings using soc_shivalik scope -> must return empty
        leaked = await list_scoped_service_requests(s, patient_id="pat_009", society_id="soc_shivalik")
        assert leaked == []

    # Document retrieval for soc_lotus must never return annex-shivalik-greens or superseded plan-coverage-v1
    lotus_docs = await toolbox.search_clinic_docs("Sunday wellness room hours", society_id="soc_lotus", k=6)
    doc_ids = {d["doc_id"] for d in lotus_docs}
    assert "annex-shivalik-greens" not in doc_ids
    assert "annex-royal-palms" not in doc_ids
    assert "annex-lotus-tower" in doc_ids

    # Version conflict check: plan-coverage-v2 supersedes v1 (Thyroid covered, Vitamin D not covered)
    cov_docs = await toolbox.search_clinic_docs("Thyroid Profile Vitamin D covered", society_id="soc_lotus", k=6)
    top_text = " ".join(d["content"] for d in cov_docs)
    assert "Thyroid Profile is now covered" in top_text
    assert "Vitamin D (25-OH) is no longer covered" in top_text
    assert "Platinum Care" not in top_text