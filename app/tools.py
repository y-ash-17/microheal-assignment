import uuid
from datetime import datetime, timezone
from typing import Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.booking import BookingConflictError, create_or_get_booking
from app.classifier import Service
from app.models import Escalation, Patient, ServiceRequest, Society, SymptomLog
from app.rag import retrieve_corpus_chunks
from app.safety import SafetyAssessment, assess_safety
from app.scoping import list_scoped_service_requests, verify_patient_in_society


class BookServiceArgs(BaseModel):
    patient_id: str = Field(..., min_length=1)
    service: Service
    subtype: str = Field(..., min_length=1)
    scheduled_for: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)


class ListServiceRequestsArgs(BaseModel):
    patient_id: str = Field(..., min_length=1)
    status: str | None = Field(default=None)


class LogSymptomArgs(BaseModel):
    patient_id: str = Field(..., min_length=1)
    symptom: str = Field(..., min_length=1)
    severity: int = Field(..., ge=1, le=5)
    request_id: str = Field(..., min_length=1)


class SearchClinicDocsArgs(BaseModel):
    query: str = Field(..., min_length=1)
    society_id: str = Field(..., min_length=1)
    k: int = Field(default=6, ge=1, le=20)


class EscalateEmergencyArgs(BaseModel):
    patient_id: str = Field(..., min_length=1)
    transcript: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)


async def _open_session(session_or_factory: Any):
    if isinstance(session_or_factory, AsyncSession) or hasattr(session_or_factory, "execute"):
        return session_or_factory, None
    cand = session_or_factory() if callable(session_or_factory) else session_or_factory
    if hasattr(cand, "__aenter__"):
        entered = await cand.__aenter__()
        return entered, ("aexit", cand)
    if hasattr(cand, "__aiter__"):
        item = await cand.__anext__()
        if hasattr(item, "__aenter__"):
            entered = await item.__aenter__()
            return entered, ("gen_aexit", cand, item)
        return item, ("gen", cand)
    return cand, None


async def _close_session(cleanup: Any) -> None:
    if not cleanup:
        return
    kind = cleanup[0]
    try:
        if kind == "aexit":
            await cleanup[1].__aexit__(None, None, None)
        elif kind == "gen_aexit":
            await cleanup[2].__aexit__(None, None, None)
            await cleanup[1].aclose()
        elif kind == "gen":
            await cleanup[1].aclose()
    except Exception:
        pass

async def _resolve_clinician_for_slot(
    db: AsyncSession,
    *,
    society_id: str,
    service: str,
    subtype: str,
    scheduled_for: str,
) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from app.models import Clinician

    dt = datetime.fromisoformat(scheduled_for)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    dt_local = dt.astimezone(ZoneInfo("Asia/Kolkata"))
    weekday_full = dt_local.strftime("%A").lower()
    weekday_short3 = weekday_full[:3]
    mins = dt_local.hour * 60 + dt_local.minute

    clinicians = (await db.execute(select(Clinician))).scalars().all()
    sub_upper = (subtype or "").upper()
    svc_upper = (service or "").upper()

    for c in clinicians:
        if c.service.upper() != svc_upper:
            continue
        subtypes = [str(s).upper() for s in (c.subtypes or [])]
        if sub_upper and subtypes and sub_upper not in subtypes:
            continue
        step = int(c.slot_minutes or 20)
        for win in (c.availability or []):
            if not isinstance(win, dict):
                continue
            if win.get("society_id") != society_id:
                continue
            w_day = str(win.get("day") or win.get("weekday") or "").strip().lower()
            if w_day not in (weekday_full, weekday_short3):
                continue
            sh, sm = [int(x) for x in str(win.get("start", "00:00")).split(":")[:2]]
            eh, em = [int(x) for x in str(win.get("end", "23:59")).split(":")[:2]]
            s_m, e_m = sh * 60 + sm, eh * 60 + em
            if s_m <= mins and (mins + step) <= e_m and (mins - s_m) % step == 0:
                return c.id

    # Fallback to matched clinician for this society, service, and specialty
    for c in clinicians:
        if c.service.upper() != svc_upper:
            continue
        subtypes = [str(s).upper() for s in (c.subtypes or [])]
        if sub_upper and subtypes and sub_upper not in subtypes:
            continue
        avail = c.availability if isinstance(c.availability, list) else []
        if any(isinstance(w, dict) and w.get("society_id") == society_id for w in avail):
            return c.id

    raise ValueError(
        f"No clinician found for service '{svc_upper}' and specialty '{sub_upper}' "
        f"available at society '{society_id}' for slot '{scheduled_for}'."
    )


async def _exec_book_service(
    session_or_factory: Any,
    *,
    patient_id: str,
    service: str,
    subtype: str,
    scheduled_for: str,
    request_id: str,
    thread_id: str = "default_thread",
    clinician_id: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    db, cleanup = await _open_session(session_or_factory)
    try:
        patient = await db.get(Patient, patient_id)
        if not patient:
            raise ValueError(f"Unknown patient_id: {patient_id}")

        svc_val = Service(service.upper()).value if service else Service.DOCTOR.value
        resolved_clinician = clinician_id or await _resolve_clinician_for_slot(
            db,
            society_id=patient.society_id,
            service=svc_val,
            subtype=subtype,
            scheduled_for=scheduled_for,
        )

        return await create_or_get_booking(
            db,
            request_id=request_id,
            thread_id=thread_id,
            patient_id=patient_id,
            clinician_id=resolved_clinician,
            service=svc_val,
            subtype=subtype.upper(),
            scheduled_for=scheduled_for,
            notes=notes,
        )
    finally:
        await _close_session(cleanup)


async def _exec_list_service_requests(
    db: AsyncSession,
    *,
    patient_id: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Unknown patient_id: {patient_id}")

    rows = await list_scoped_service_requests(
        db,
        society_id=patient.society_id,
        patient_id=patient_id,
        status=status,
    )
    return [
        {
            "id": r.id,
            "request_id": r.request_id,
            "thread_id": r.thread_id,
            "patient_id": r.patient_id,
            "society_id": r.society_id,
            "clinician_id": r.clinician_id,
            "service": r.service,
            "subtype": r.subtype,
            "scheduled_for": r.scheduled_for,
            "slot_start": r.scheduled_for,
            "status": r.status,
        }
        for r in rows
    ]


async def _exec_log_symptom(
    db: AsyncSession,
    *,
    patient_id: str,
    symptom: str,
    severity: int,
    request_id: str,
    thread_id: str = "default_thread",
) -> dict[str, Any]:
    if not isinstance(severity, int) or isinstance(severity, bool) or severity < 1 or severity > 5:
        raise ValueError("Severity must be an integer between 1 and 5.")

    existing = (
        await db.execute(select(SymptomLog).where(SymptomLog.request_id == request_id))
    ).scalar_one_or_none()
    if existing:
        return {
            "id": existing.id,
            "request_id": existing.request_id,
            "patient_id": existing.patient_id,
            "society_id": existing.society_id,
            "symptom": existing.symptom,
            "severity": existing.severity,
            "idempotent_replay": True,
        }

    patient = await db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Unknown patient_id: {patient_id}")

    log = SymptomLog(
        id=f"sym_{uuid.uuid4().hex[:12]}",
        request_id=request_id,
        thread_id=thread_id,
        patient_id=patient.id,
        society_id=patient.society_id,
        symptom=symptom,
        severity=severity,
    )
    db.add(log)
    try:
        await db.commit()
    except IntegrityError:
        # A concurrent retry can pass the pre-insert lookup.  The database
        # uniqueness rule is the authority; after it rejects our insert, read
        # the original result and return the normal idempotent response.
        await db.rollback()
        replay = (
            await db.execute(select(SymptomLog).where(SymptomLog.request_id == request_id))
        ).scalar_one_or_none()
        if replay is None:
            raise
        if replay.patient_id != patient_id:
            raise ValueError("Request ID belongs to another patient.")
        return {
            "id": replay.id,
            "request_id": replay.request_id,
            "patient_id": replay.patient_id,
            "society_id": replay.society_id,
            "symptom": replay.symptom,
            "severity": replay.severity,
            "idempotent_replay": True,
        }
    await db.refresh(log)
    return {
        "id": log.id,
        "request_id": log.request_id,
        "patient_id": log.patient_id,
        "society_id": log.society_id,
        "symptom": log.symptom,
        "severity": log.severity,
        "idempotent_replay": False,
    }


async def _exec_search_clinic_docs(
    db: AsyncSession,
    *,
    query: str,
    society_id: str,
    k: int = 6,
) -> list[dict[str, Any]]:
    if not society_id:
        raise ValueError("society_id is required.")
    society = await db.get(Society, society_id)
    res = await retrieve_corpus_chunks(
        db,
        query,
        society=society,
        society_id=society_id,
        top_k=k,
        include_internal=False,
    )
    if res.get("abstained"):
        return []
    return res.get("chunks", [])


async def _exec_escalate_emergency(
    db: AsyncSession,
    *,
    patient_id: str,
    transcript: str,
    request_id: str,
    thread_id: str = "default_thread",
    level: int = 1,
    sla_minutes: int = 5,
) -> dict[str, Any]:
    existing = (
        await db.execute(select(Escalation).where(Escalation.request_id == request_id))
    ).scalar_one_or_none()
    if existing:
        return {
            "escalation_id": existing.id,
            "id": existing.id,
            "request_id": existing.request_id,
            "patient_id": existing.patient_id,
            "society_id": existing.society_id,
            "level": existing.level,
            "sla_minutes": existing.sla_minutes,
            "status": existing.status,
            "call_112": existing.level == 1,
            "idempotent_replay": True,
            "nurse_notified": True,
        }

    patient = await db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Unknown patient_id: {patient_id}")

    society = await db.get(Society, patient.society_id)
    emergency_contact = ""
    if society and isinstance(society.hospital, dict):
        emergency_contact = f"{society.hospital.get('name', '')} {society.hospital.get('emergency_phone', '')}".strip()

    esc = Escalation(
        id=f"esc_{uuid.uuid4().hex[:12]}",
        request_id=request_id,
        thread_id=thread_id,
        patient_id=patient.id,
        society_id=patient.society_id,
        level=level,
        sla_minutes=sla_minutes,
        status="OPEN",
        trigger_message=transcript,
        emergency_contact=emergency_contact,
    )
    db.add(esc)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        replay = (
            await db.execute(select(Escalation).where(Escalation.request_id == request_id))
        ).scalar_one_or_none()
        if replay is None:
            raise
        if replay.patient_id != patient_id:
            raise ValueError("Request ID belongs to another patient.")
        return {
            "escalation_id": replay.id,
            "id": replay.id,
            "request_id": replay.request_id,
            "patient_id": replay.patient_id,
            "society_id": replay.society_id,
            "level": replay.level,
            "sla_minutes": replay.sla_minutes,
            "status": replay.status,
            "call_112": replay.level == 1,
            "idempotent_replay": True,
            "nurse_notified": True,
        }
    await db.refresh(esc)

    from app.notifications import dispatch_nurse_escalation_alert
    notif = await dispatch_nurse_escalation_alert(
        escalation_id=esc.id,
        patient_id=patient.id,
        society_id=patient.society_id,
        level=level,
        sla_minutes=sla_minutes,
        trigger_message=transcript,
        patient_name=patient.name,
        patient_phone=patient.phone,
        flat=patient.flat,
        emergency_contact=emergency_contact,
    )

    return {
        "escalation_id": esc.id,
        "id": esc.id,
        "request_id": esc.request_id,
        "patient_id": esc.patient_id,
        "society_id": esc.society_id,
        "level": esc.level,
        "sla_minutes": esc.sla_minutes,
        "status": esc.status,
        "emergency_contact": esc.emergency_contact,
        "call_112": esc.level == 1,
        "idempotent_replay": False,
        "nurse_notified": notif.get("nurse_notified", True),
        "notification_channel": notif.get("channel", "WELLNESS_ROOM_DESK"),
        "alert_dispatched_at": notif.get("dispatched_at"),
    }


@tool("book_service", args_schema=BookServiceArgs)
async def book_service(
    patient_id: str,
    service: Service,
    subtype: str,
    scheduled_for: str,
    request_id: str,
) -> dict[str, Any]:
    """Book a doctor consultation or lab test for a patient against a slot."""
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await _exec_book_service(
            db,
            patient_id=patient_id,
            service=service,
            subtype=subtype,
            scheduled_for=scheduled_for,
            request_id=request_id,
        )


@tool("list_service_requests", args_schema=ListServiceRequestsArgs)
async def list_service_requests(
    patient_id: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List service requests/bookings for a patient within their society."""
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await _exec_list_service_requests(
            db,
            patient_id=patient_id,
            status=status,
        )


@tool("log_symptom", args_schema=LogSymptomArgs)
async def log_symptom(
    patient_id: str,
    symptom: str,
    severity: int,
    request_id: str,
) -> dict[str, Any]:
    """Log a patient symptom with severity 1..5."""
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await _exec_log_symptom(
            db,
            patient_id=patient_id,
            symptom=symptom,
            severity=severity,
            request_id=request_id,
        )


@tool("search_clinic_docs", args_schema=SearchClinicDocsArgs)
async def search_clinic_docs(
    query: str,
    society_id: str,
    k: int = 6,
) -> list[dict[str, Any]]:
    """Search active clinic policies and society annex documents."""
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await _exec_search_clinic_docs(
            db,
            query=query,
            society_id=society_id,
            k=k,
        )


@tool("escalate_emergency", args_schema=EscalateEmergencyArgs)
async def escalate_emergency(
    patient_id: str,
    transcript: str,
    request_id: str,
) -> dict[str, Any]:
    """Record an emergency escalation row for a patient."""
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await _exec_escalate_emergency(
            db,
            patient_id=patient_id,
            transcript=transcript,
            request_id=request_id,
        )


CONTRACT_TOOLS = [
    book_service,
    list_service_requests,
    log_symptom,
    search_clinic_docs,
    escalate_emergency,
]


class CareConciergeToolbox:
    """Service-layer toolbox wrapping the 5 contract actions and safety checks."""

    def __init__(self, session_or_factory: Any = None, session: AsyncSession | None = None, session_factory=None):
        if session is not None or session_factory is not None:
            self.session = session
            self.session_factory = session_factory
        elif session_or_factory is not None:
            if isinstance(session_or_factory, AsyncSession) or hasattr(session_or_factory, "execute"):
                self.session = session_or_factory
                self.session_factory = None
            else:
                self.session = None
                self.session_factory = session_or_factory
        else:
            self.session = None
            self.session_factory = None

    def check_safety(self, message: str, classifier_fn=None) -> SafetyAssessment:
        return assess_safety(message, classifier_fn=classifier_fn)

    async def book_service(
        self,
        patient_id: str,
        service: str,
        subtype: str,
        scheduled_for: str,
        request_id: str,
        *,
        thread_id: str = "default_thread",
        clinician_id: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        try:
            return await _exec_book_service(
                self.session_factory or self.session,
                patient_id=patient_id,
                service=service,
                subtype=subtype,
                scheduled_for=scheduled_for,
                request_id=request_id,
                thread_id=thread_id,
                clinician_id=clinician_id,
                notes=notes,
            )
        except BookingConflictError as exc:
            return {
                "status": "SLOT_TAKEN",
                "error": str(exc),
                "request_id": request_id,
                "patient_id": patient_id,
                "scheduled_for": scheduled_for,
            }

    async def search_clinic_docs(
        self,
        query: str,
        society_id: str,
        k: int = 6,
    ) -> list[dict[str, Any]]:
        db, cleanup = await _open_session(self.session_factory or self.session)
        try:
            society = await db.get(Society, society_id)
            res = await retrieve_corpus_chunks(
                db,
                query,
                society=society,
                society_id=society_id,
                top_k=k,
                include_internal=False,
            )
            return res.get("chunks") or []
        finally:
            await _close_session(cleanup)
            
    async def list_service_requests(
        self,
        patient_id: str,
        status: str | None = None,
        *,
        session: Any = None,
    ) -> list[dict[str, Any]]:
        db, cleanup = await _open_session(session or self.session or self.session_factory)
        try:
            return await _exec_list_service_requests(db, patient_id=patient_id, status=status)
        finally:
            await _close_session(cleanup)

    async def log_symptom(
        self,
        patient_id: str,
        symptom: str,
        severity: int,
        request_id: str,
        *,
        thread_id: str = "default_thread",
        session: Any = None,
    ) -> dict[str, Any]:
        db, cleanup = await _open_session(session or self.session or self.session_factory)
        try:
            return await _exec_log_symptom(
                db,
                patient_id=patient_id,
                symptom=symptom,
                severity=severity,
                request_id=request_id,
                thread_id=thread_id,
            )
        finally:
            await _close_session(cleanup)

    
    async def escalate_emergency(
        self,
        patient_id: str,
        transcript: str,
        request_id: str,
        *,
        thread_id: str = "default_thread",
        level: int = 1,
        sla_minutes: int = 5,
        session: Any = None,
    ) -> dict[str, Any]:
        db, cleanup = await _open_session(session or self.session or self.session_factory)
        try:
            return await _exec_escalate_emergency(
                db,
                patient_id=patient_id,
                transcript=transcript,
                request_id=request_id,
                thread_id=thread_id,
                level=level,
                sla_minutes=sla_minutes,
            )
        finally:
            await _close_session(cleanup)


__all__ = [
    "BookServiceArgs",
    "ListServiceRequestsArgs",
    "LogSymptomArgs",
    "SearchClinicDocsArgs",
    "EscalateEmergencyArgs",
    "book_service",
    "list_service_requests",
    "log_symptom",
    "search_clinic_docs",
    "escalate_emergency",
    "CONTRACT_TOOLS",
    "CareConciergeToolbox",
    "BookingConflictError",
]
