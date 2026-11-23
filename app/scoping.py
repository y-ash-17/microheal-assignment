from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Clinician, DocChunk, Patient, ServiceRequest, Society


def _normalize_society_id(society_or_id: Society | str | None) -> str | None:
    if society_or_id is None:
        return None
    if isinstance(society_or_id, Society):
        return society_or_id.id
    return str(society_or_id).strip() or None


async def verify_patient_in_society(
    session: AsyncSession,
    *,
    patient_id: str,
    society_id: str,
) -> Patient:
    if not society_id:
        raise ValueError("society_id is required")
    patient = await session.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Unknown patient_id: {patient_id}")
    if patient.society_id != society_id:
        raise PermissionError(
            f"Patient {patient_id} belongs to society {patient.society_id}, not {society_id}."
        )
    return patient


async def list_scoped_service_requests(
    session: AsyncSession,
    *,
    society_id: str,
    patient_id: str | None = None,
    status: str | None = None,
) -> list[ServiceRequest]:
    """Read service_requests strictly scoped to a required society_id."""
    if not society_id:
        raise ValueError("society_id is required when reading service requests.")

    stmt = select(ServiceRequest).where(ServiceRequest.society_id == society_id)
    if patient_id is not None:
        stmt = stmt.where(ServiceRequest.patient_id == patient_id)
    if status is not None:
        stmt = stmt.where(ServiceRequest.status == status)

    rows = (await session.execute(stmt)).scalars().all()
    # Defense-in-depth filter to guarantee zero cross-society rows
    return [r for r in rows if r.society_id == society_id]


def scoped_document_conditions(*, society_id: str, include_internal: bool = False) -> list:
    """Return the mandatory SQL predicates for every patient document read."""
    if not society_id:
        raise ValueError("society_id is required when reading documents.")

    conditions = [
        DocChunk.superseded_by.is_(None),
        or_(DocChunk.applies_to == "all", DocChunk.applies_to == society_id),
    ]
    if not include_internal:
        conditions.extend(
            [
                DocChunk.doc_id != "staff-notes-internal",
                ~DocChunk.doc_id.ilike("%internal%"),
            ]
        )
    return conditions


def filter_scoped_chunks(
    chunks: list[DocChunk],
    *,
    society_id: str,
    include_internal: bool = False,
) -> list[DocChunk]:
    """Filter document chunks to active (non-superseded) items applicable to society_id.

    By default, internal documents (such as staff-notes-internal) are strictly excluded
    from patient concierge retrieval.
    """
    if not society_id:
        raise ValueError("society_id is required when filtering documents.")

    soc_id = society_id.strip()
    annex_map = {
        "soc_lotus": "annex-lotus-tower",
        "soc_shivalik": "annex-shivalik-greens",
        "soc_royal": "annex-royal-palms",
    }
    allowed_annex = annex_map.get(soc_id)

    scoped: list[DocChunk] = []
    for ch in chunks:
        if getattr(ch, "superseded_by", None) is not None:
            continue
        doc_l = (getattr(ch, "doc_id", "") or "").lower()

        # Strict defense-in-depth: exclude internal staff notes from patient queries
        if not include_internal and ("internal" in doc_l or doc_l == "staff-notes-internal"):
            continue

        applies = (getattr(ch, "applies_to", "all") or "all").strip()

        if doc_l.startswith("annex-"):
            if allowed_annex and getattr(ch, "doc_id", "") != allowed_annex:
                continue
            if applies not in ("all", soc_id, allowed_annex):
                continue
        else:
            if applies not in ("all", soc_id):
                continue

        scoped.append(ch)
    return scoped


def list_scoped_clinicians(
    clinicians: list[Clinician],
    *,
    society: Society | None = None,
    society_id: str | None = None,
) -> list[Clinician]:
    """Filter clinicians available at the given society's partner hospital or availability windows."""
    target_soc_id = _normalize_society_id(society or society_id)
    hospital_name = ""
    if society and isinstance(society.hospital, dict):
        hospital_name = (society.hospital.get("name") or "").strip().lower()

    matched: list[Clinician] = []
    for c in clinicians:
        avail = c.availability if isinstance(c.availability, list) else []
        has_soc_window = any(
            isinstance(w, dict) and w.get("society_id") == target_soc_id
            for w in avail
            if target_soc_id
        )
        hosp_match = bool(hospital_name and hospital_name in (c.hospital or "").lower())
        if has_soc_window or hosp_match or not target_soc_id:
            matched.append(c)
    return matched
