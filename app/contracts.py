from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ServiceEnum(str, Enum):
    DOCTOR = "DOCTOR"
    LAB = "LAB"


class SubtypeEnum(str, Enum):
    GENERAL_PHYSICIAN = "GENERAL_PHYSICIAN"
    CARDIOLOGY = "CARDIOLOGY"
    PAEDIATRICS = "PAEDIATRICS"
    ORTHOPAEDICS = "ORTHOPAEDICS"
    DERMATOLOGY = "DERMATOLOGY"
    ENT = "ENT"
    GYNAECOLOGY = "GYNAECOLOGY"
    PHYSIOTHERAPY = "PHYSIOTHERAPY"
    CBC = "CBC"
    FASTING_BLOOD_SUGAR = "FASTING_BLOOD_SUGAR"
    HBA1C = "HBA1C"
    LIPID_PROFILE = "LIPID_PROFILE"
    LFT = "LFT"
    KFT = "KFT"
    THYROID_PROFILE = "THYROID_PROFILE"
    VITAMIN_D = "VITAMIN_D"
    URINE_ROUTINE = "URINE_ROUTINE"


class BookingStatusEnum(str, Enum):
    CONFIRMED = "CONFIRMED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    NO_SHOW = "NO_SHOW"


Service = Literal["DOCTOR", "LAB"]
Intent = Literal[
    "BOOK_SERVICE",
    "LIST_SERVICES",
    "LOG_SYMPTOM",
    "CLINIC_INFO",
    "MEDICAL_ADVICE",
    "EMERGENCY",
    "SMALL_TALK",
    "UNKNOWN",
]
SafetyVerdict = Literal["EMERGENCY", "MEDICAL_ADVICE", "SAFE"]


class BookServiceArgs(BaseModel):
    patient_id: str = Field(..., description="Patient identifier, e.g. pat_001")
    service: ServiceEnum = Field(..., description="DOCTOR or LAB")
    subtype: SubtypeEnum = Field(..., description="Department or lab test code from directory")
    scheduled_for: str = Field(..., description="ISO-8601 datetime with timezone offset, e.g. 2027-01-12T10:30:00+05:30")
    request_id: str = Field(..., description="Client idempotency UUID for this turn")


class ListServiceRequestsArgs(BaseModel):
    patient_id: str = Field(..., description="Patient identifier, e.g. pat_001")
    status: Optional[BookingStatusEnum] = Field(default=None)


class LogSymptomArgs(BaseModel):
    patient_id: str = Field(..., description="Patient identifier, e.g. pat_001")
    symptom: str = Field(..., min_length=1)
    severity: int = Field(..., ge=1, le=5)
    request_id: str = Field(..., description="Client idempotency UUID for this turn")


class SearchClinicDocsArgs(BaseModel):
    query: str = Field(..., min_length=1)
    society_id: str = Field(...)
    k: int = Field(default=6, ge=1, le=12)


class EscalateEmergencyArgs(BaseModel):
    patient_id: str = Field(..., description="Patient identifier, e.g. pat_001")
    transcript: str = Field(..., min_length=1)
    request_id: str = Field(..., description="Client idempotency UUID for this turn")


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)
    language: Optional[str] = Field(default=None) 