"""Enforce idempotency for symptom and emergency write tools.

Revision ID: 002_write_idempotency
Revises: 001_initial_schema
Create Date: 2026-11-22
"""

from alembic import op


revision = "002_write_idempotency"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_symptom_logs_request_id", "symptom_logs", ["request_id"]
    )
    op.create_unique_constraint(
        "uq_escalations_request_id", "escalations", ["request_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_escalations_request_id", "escalations", type_="unique")
    op.drop_constraint("uq_symptom_logs_request_id", "symptom_logs", type_="unique")
