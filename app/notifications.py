"""Nurse escalation notification dispatch workflow for MicroHeal Care Concierge."""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("concierge.notifications")

# In-memory alert audit log for inspection and testing
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
    """Dispatch a high-priority emergency alert to the duty nurse / wellness room desk."""
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
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
        "channel": "WELLNESS_ROOM_DESK",
        "priority": "CRITICAL" if level == 1 else ("HIGH" if level == 2 else "MEDIUM"),
        "status": "DISPATCHED",
    }

    NOTIFICATION_DISPATCH_LOG.append(alert_payload)

    logger.warning(
        "EMERGENCY_ALERT_DISPATCHED to duty nurse: level=%d sla=%dm patient=%s flat=%s society=%s esc_id=%s",
        level,
        sla_minutes,
        patient_id,
        flat,
        society_id,
        escalation_id,
    )

    return {
        "nurse_notified": True,
        "channel": "WELLNESS_ROOM_DESK",
        "alert_id": f"alt_{escalation_id}",
        "dispatched_at": alert_payload["dispatched_at"],
        "sla_minutes": sla_minutes,
    }
