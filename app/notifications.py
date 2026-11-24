"""Local escalation-alert audit workflow for MicroHeal Care Concierge.

This module deliberately does not contact a nurse.  The assignment requires an
emergency to create a stored escalation row; it does not provide a pager, SMS,
WhatsApp, or hospital integration.  We therefore record an auditable alert
payload in-process and return an explicit ``nurse_notified: False`` result so
the chat response and API never overstate what happened.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("concierge.notifications")

# Process-local audit log for inspection and tests.  It is not durable: the
# durable record of the emergency is the ``escalations`` row in PostgreSQL.
NOTIFICATION_DISPATCH_LOG: list[dict[str, Any]] = []


async def dispatch_nurse_escalation_alert(
    *,
    escalation_id: str,
    patient_id: str,
    society_id: str,
    level: int,
    sla_minutes: int,
    trigger_message: str,
    patient_name: str = "",
    patient_phone: str = "",
    flat: str = "",
    emergency_contact: str = "",
) -> dict[str, Any]:
    """Record a local alert entry for an escalation; do not contact a nurse.

    The return value is intentionally explicit about the missing delivery:
    ``nurse_notified`` is always ``False`` and ``delivery_status`` is
    ``NOT_SENT``.  A real deployment would replace this function with a pager
    or hospital-HIS adapter and only then report successful notification.
    """
    recorded_at = datetime.now(timezone.utc).isoformat()
    alert_payload = {
        "escalation_id": escalation_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_phone": patient_phone,
        "flat": flat,
        "society_id": society_id,
        "level": level,
        "sla_minutes": sla_minutes,
        "trigger_message": trigger_message,
        "emergency_contact": emergency_contact,
        "recorded_at": recorded_at,
        "channel": "LOCAL_AUDIT_LOG",
        "priority": "CRITICAL" if level == 1 else ("HIGH" if level == 2 else "MEDIUM"),
        "delivery_status": "NOT_SENT",
        "status": "RECORDED_NOT_SENT",
    }

    NOTIFICATION_DISPATCH_LOG.append(alert_payload)

    logger.warning(
        "ESCALATION_ALERT_RECORDED_LOCALLY (nurse_notified=false): level=%d sla=%dm "
        "patient=%s flat=%s society=%s esc_id=%s",
        level,
        sla_minutes,
        patient_id,
        flat,
        society_id,
        escalation_id,
    )

    return {
        "nurse_notified": False,
        "alert_recorded": True,
        "channel": "LOCAL_AUDIT_LOG",
        "delivery_status": "NOT_SENT",
        "alert_id": f"alt_{escalation_id}",
        "recorded_at": recorded_at,
        "sla_minutes": sla_minutes,
        "note": (
            "No external nurse paging is configured in this demo. The escalation "
            "is stored in the database and this local audit entry was recorded."
        ),
    }
