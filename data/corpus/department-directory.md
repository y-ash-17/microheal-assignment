---
doc_id: department-directory
title: Department and Test Directory
applies_to: all
version: 4
effective_from: 2026-03-01
superseded_by: null
---

# Department and Test Directory

Codes on this page are the values the booking system uses. The `subtype` of a booking must be one of them. Which clinicians visit your society is listed in your society's annex.

## Doctor departments

| Code | Department | Hindi | Slot length | Seen at |
|---|---|---|---|---|
| `GENERAL_PHYSICIAN` | General Physician (OPD-1) | सामान्य चिकित्सक | 20 min | Wellness room |
| `CARDIOLOGY` | Cardiology (OPD-4) | हृदय रोग | 30 min | Wellness room |
| `PAEDIATRICS` | Paediatrics (OPD-2) | बाल रोग | 20 min | Wellness room |
| `ORTHOPAEDICS` | Orthopaedics (OPD-6) | हड्डी रोग | 30 min | Wellness room |
| `DERMATOLOGY` | Dermatology (OPD-7, "DERM") | त्वचा रोग | 30 min | Wellness room |
| `ENT` | Ear, Nose and Throat (OPD-5) | नाक, कान, गला | 30 min | Wellness room |
| `GYNAECOLOGY` | Obstetrics and Gynaecology (OPD-3) | स्त्री रोग | 30 min | Wellness room |
| `PHYSIOTHERAPY` | Physiotherapy (REHAB-1) | फिजियोथेरेपी | 45 min | At home |

Physiotherapy is booked as a `DOCTOR` service with subtype `PHYSIOTHERAPY`. It is only offered where the society annex lists it.

## Lab tests

All lab tests are booked as a `LAB` service and collected at home.

| Code | Test | Sample | Fasting | Report in |
|---|---|---|---|---|
| `CBC` | Complete Blood Count | Blood (EDTA) | No | 6 hours |
| `FASTING_BLOOD_SUGAR` | Fasting Blood Sugar (FBS) | Blood (fluoride) | 8–10 h | 12 hours |
| `HBA1C` | HbA1c (glycated haemoglobin) | Blood (EDTA) | No | 24 hours |
| `LIPID_PROFILE` | Lipid Profile | Blood (serum) | 10–12 h | 24 hours |
| `LFT` | Liver Function Test | Blood (serum) | No | 24 hours |
| `KFT` | Kidney Function Test (RFT) | Blood (serum) | No | 24 hours |
| `THYROID_PROFILE` | Thyroid Profile (T3, T4, TSH) | Blood (serum) | No | 24 hours |
| `VITAMIN_D` | Vitamin D (25-OH) | Blood (serum) | No | 48 hours |
| `URINE_ROUTINE` | Urine Routine and Microscopy | Urine | No | 12 hours |

## Common alternative names

- "Sugar test" usually means `FASTING_BLOOD_SUGAR`; "3-month sugar" means `HBA1C`.
- "Cholesterol test" means `LIPID_PROFILE`.
- "TFT" and "TSH test" mean `THYROID_PROFILE`.
- "RFT" and "kidney test" mean `KFT`.
- "Skin doctor" means `DERMATOLOGY`; "heart doctor" means `CARDIOLOGY`; "bone doctor" means `ORTHOPAEDICS`.
