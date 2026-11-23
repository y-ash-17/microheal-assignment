from datetime import datetime, timezone
from typing import Any
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Society(Base):
    __tablename__ = "societies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    address: Mapped[str] = mapped_column(String(300), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata", nullable=False)
    annex_doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    hospital: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    patients: Mapped[list["Patient"]] = relationship("Patient", back_populates="society")


class Clinician(Base):
    __tablename__ = "clinicians"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    service: Mapped[str] = mapped_column(String(64), nullable=False)  # DOCTOR | LAB
    subtypes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    hospital: Mapped[str] = mapped_column(String(200), nullable=False)
    slot_minutes: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    availability: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    society_id: Mapped[str] = mapped_column(ForeignKey("societies.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(32), default="en", nullable=False)
    date_of_birth: Mapped[str] = mapped_column(String(32), nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    flat: Mapped[str] = mapped_column(String(64), nullable=False)

    society: Mapped["Society"] = relationship("Society", back_populates="patients")


class ServiceRequest(Base):
    __tablename__ = "service_requests"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_service_requests_request_id"),
        Index(
            "uq_service_requests_active_clinician_slot",
            "clinician_id",
            "scheduled_for",
            unique=True,
            postgresql_where=text("status IN ('CONFIRMED', 'PENDING')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), default="default_thread", nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    society_id: Mapped[str] = mapped_column(ForeignKey("societies.id"), nullable=False, index=True)
    clinician_id: Mapped[str] = mapped_column(ForeignKey("clinicians.id"), nullable=False, index=True)
    service: Mapped[str] = mapped_column(String(64), nullable=False)  # DOCTOR | LAB
    subtype: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    scheduled_for: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="CONFIRMED", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SymptomLog(Base):
    __tablename__ = "symptom_logs"
    __table_args__ = (
        CheckConstraint("severity >= 1 AND severity <= 5", name="ck_symptom_logs_severity_1_5"),
        UniqueConstraint("request_id", name="uq_symptom_logs_request_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), default="default_thread", nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    society_id: Mapped[str] = mapped_column(ForeignKey("societies.id"), nullable=False, index=True)
    symptom: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    def __init__(self, **kwargs: Any):
        if "request_id" not in kwargs or not kwargs["request_id"]:
            kwargs["request_id"] = kwargs.get("thread_id") or "sym_req_default"
        super().__init__(**kwargs)


class Escalation(Base):
    __tablename__ = "escalations"
    __table_args__ = (
        CheckConstraint("level >= 1 AND level <= 5", name="ck_escalations_level_1_5"),
        UniqueConstraint("request_id", name="uq_escalations_request_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), default="", nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    society_id: Mapped[str] = mapped_column(ForeignKey("societies.id"), nullable=False, index=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    sla_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="OPEN", nullable=False)
    trigger_message: Mapped[str] = mapped_column(Text, nullable=False)
    emergency_contact: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    applies_to: Mapped[str] = mapped_column(String(128), default="all", nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    effective_from: Mapped[str | None] = mapped_column(String(32), nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    chunks: Mapped[list["DocChunk"]] = relationship(
        "DocChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocChunk(Base):
    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    section: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    heading_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    applies_to: Mapped[str] = mapped_column(String(128), default="all", nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    effective_from: Mapped[str | None] = mapped_column(String(32), nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")


Booking = ServiceRequest
CorpusChunk = DocChunk
