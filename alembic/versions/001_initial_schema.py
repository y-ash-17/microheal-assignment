"""Initial 8-table schema for MicroHeal Care Concierge

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. societies
    op.create_table(
        "societies",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("address", sa.String(length=300), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False, server_default="Asia/Kolkata"),
        sa.Column("annex_doc_id", sa.String(length=128), nullable=False),
        sa.Column("hospital", sa.JSON(), nullable=False),
    )

    # 2. clinicians
    op.create_table(
        "clinicians",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("service", sa.String(length=64), nullable=False),
        sa.Column("subtypes", sa.JSON(), nullable=False),
        sa.Column("hospital", sa.String(length=200), nullable=False),
        sa.Column("slot_minutes", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("availability", sa.JSON(), nullable=False),
    )

    # 3. patients
    op.create_table(
        "patients",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("society_id", sa.String(length=64), sa.ForeignKey("societies.id"), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("phone", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False, server_default="en"),
        sa.Column("date_of_birth", sa.String(length=32), nullable=False),
        sa.Column("gender", sa.String(length=16), nullable=False),
        sa.Column("flat", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_patients_society_id", "patients", ["society_id"])

    # 4. service_requests (with UNIQUE request_id + partial unique index on active clinician slot)
    op.create_table(
        "service_requests",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False, server_default="default_thread"),
        sa.Column("patient_id", sa.String(length=64), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("society_id", sa.String(length=64), sa.ForeignKey("societies.id"), nullable=False),
        sa.Column("clinician_id", sa.String(length=64), sa.ForeignKey("clinicians.id"), nullable=False),
        sa.Column("service", sa.String(length=64), nullable=False),
        sa.Column("subtype", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("scheduled_for", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="CONFIRMED"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("request_id", name="uq_service_requests_request_id"),
    )
    op.create_index("ix_service_requests_request_id", "service_requests", ["request_id"])
    op.create_index("ix_service_requests_thread_id", "service_requests", ["thread_id"])
    op.create_index("ix_service_requests_patient_id", "service_requests", ["patient_id"])
    op.create_index("ix_service_requests_society_id", "service_requests", ["society_id"])
    op.create_index("ix_service_requests_clinician_id", "service_requests", ["clinician_id"])
    op.create_index("ix_service_requests_scheduled_for", "service_requests", ["scheduled_for"])
    op.create_index(
        "uq_service_requests_active_clinician_slot",
        "service_requests",
        ["clinician_id", "scheduled_for"],
        unique=True,
        postgresql_where=sa.text("status IN ('CONFIRMED', 'PENDING')"),
    )

    # 5. symptom_logs (with CHECK severity 1..5)
    op.create_table(
        "symptom_logs",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False, server_default="default_thread"),
        sa.Column("patient_id", sa.String(length=64), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("society_id", sa.String(length=64), sa.ForeignKey("societies.id"), nullable=False),
        sa.Column("symptom", sa.Text(), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("severity >= 1 AND severity <= 5", name="ck_symptom_logs_severity_1_5"),
    )
    op.create_index("ix_symptom_logs_request_id", "symptom_logs", ["request_id"])
    op.create_index("ix_symptom_logs_thread_id", "symptom_logs", ["thread_id"])
    op.create_index("ix_symptom_logs_patient_id", "symptom_logs", ["patient_id"])
    op.create_index("ix_symptom_logs_society_id", "symptom_logs", ["society_id"])

    # 6. escalations
    op.create_table(
        "escalations",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("patient_id", sa.String(length=64), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("society_id", sa.String(length=64), sa.ForeignKey("societies.id"), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("sla_minutes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="OPEN"),
        sa.Column("trigger_message", sa.Text(), nullable=False),
        sa.Column("emergency_contact", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("level >= 1 AND level <= 5", name="ck_escalations_level_1_5"),
    )
    op.create_index("ix_escalations_request_id", "escalations", ["request_id"])
    op.create_index("ix_escalations_thread_id", "escalations", ["thread_id"])
    op.create_index("ix_escalations_patient_id", "escalations", ["patient_id"])
    op.create_index("ix_escalations_society_id", "escalations", ["society_id"])

    # 7. documents
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(length=128), primary_key=True, nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("applies_to", sa.String(length=128), nullable=False, server_default="all"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("effective_from", sa.String(length=32), nullable=True),
        sa.Column("superseded_by", sa.String(length=128), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
    )
    op.create_index("ix_documents_applies_to", "documents", ["applies_to"])
    op.create_index("ix_documents_superseded_by", "documents", ["superseded_by"])

    # 8. doc_chunks
    op.create_table(
        "doc_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=200), nullable=False, unique=True),
        sa.Column(
            "doc_id",
            sa.String(length=128),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("section", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("heading_path", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("applies_to", sa.String(length=128), nullable=False, server_default="all"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("effective_from", sa.String(length=32), nullable=True),
        sa.Column("superseded_by", sa.String(length=128), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
    )
    op.create_index("ix_doc_chunks_chunk_id", "doc_chunks", ["chunk_id"])
    op.create_index("ix_doc_chunks_doc_id", "doc_chunks", ["doc_id"])
    op.create_index("ix_doc_chunks_applies_to", "doc_chunks", ["applies_to"])
    op.create_index("ix_doc_chunks_superseded_by", "doc_chunks", ["superseded_by"])


def downgrade() -> None:
    op.drop_table("doc_chunks")
    op.drop_table("documents")
    op.drop_table("escalations")
    op.drop_table("symptom_logs")
    op.drop_table("service_requests")
    op.drop_table("patients")
    op.drop_table("clinicians")
    op.drop_table("societies")
