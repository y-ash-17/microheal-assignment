import re
from dataclasses import dataclass


@dataclass
class SafetyAssessment:
    level: int | None
    reason: str
    call_112: bool
    sla_minutes: int | None


LEVEL_1_PATTERNS = [
    # Breathing / Respiratory emergencies (English, Hinglish, Hindi)
    r"\bsaans\s+(?:nahi|nahin|nhi)\s+aa\s+rahi\b",
    r"\bsaans\s+(?:nahi|nahin|nhi)\s+(?:li\s+ja\s+rahi|le\s+pa\s+raha|le\s+pa\s+rahi)\b",
    r"\bsaans\s+lene\s+me\s+(?:takleef|dikkat|pareshani|problem|dard)\b",
    r"\bsaans\s+(?:fool|phool|ruk)\s+rahi\b",
    r"सांस\s+(?:नहीं\s+आ\s+रही|लेने\s+में\s+तकलीफ|फूल\s+रही|रुक\s+रही)",
    r"\bchest\s*pain\b",
    r"\bchest\s*pressure\b",
    r"\btightness\s*in\s*chest\b",
    r"\bdifficulty\s*breathing\b",
    r"\bshortness\s*of\s*breath\b",
    r"\bcan\s*'?t\s*breathe\b",
    r"\bcannot\s*breathe\b",
    r"\btrouble\s*breathing\b",

    # Cardiac & Chest in Hindi / Hinglish
    r"\b(?:chhati|chhaati|sine|seene)\s+me\s+.*(?:dard|jalan|pressure|jakdan)\b",
    r"\b(?:dil\s+ka\s+daura|heart\s*attack)\b",
    r"(?:सीने|छाती)\s+में\s+.*(?:दर्द|जलन|दबाव)",
    r"दिल\s+का\s+दौरा",

    # Neurological / Consciousness / Collapse
    r"\bunconscious\b",
    r"\bcollapsed\b",
    r"\bnot\s*responding\b",
    r"\bpassed\s*out\b",
    r"\b(?:behosh|behoshi|hosh\s+kho\s+baitha|gira\s+pada)\b",
    r"बेहोश",

    # Bleeding & Hemorrhage
    r"\bsevere\s*bleeding\b",
    r"\bheavy\s*bleeding\b",
    r"\b(?:blee+ding|khoon)\b.*(?:ruk\s+(?:nahi|nahin|nhi)|tez|zyada)",
    r"\b(?:ruk\s+(?:nahi|nahin|nhi)\s+rahi)\b",
    r"\b(?:khoon\s+beh\s+raha|khoon\s+ki\s+ulti)\b",
    r"खून\s+(?:बह\s+रहा|की\s+उल्टी)",

    # Stroke / Seizure
    r"\bweakness\s*(of|in)?\s*(the\s*)?(face|arm|leg)\b",
    r"\bslurred\s*speech\b",
    r"\bseizure\b",
    r"\b(?:lakwa|falij|mirgi|daura\s+pad)\b",
    r"(?:लकवा|दौरा)",

    # Self-Harm & Suicide
    r"\bsuicid(al|e)\b",
    r"\bend(ing)?\s*(my|their|his|her)?\s*life\b",
    r"\bwant\s*to\s*die\b",
    r"\b(?:marne\s+ka\s+man|jaan\s+dena\s+chahta|atmahatya)\b",
    r"आत्महत्या",
]

LEVEL_2_PATTERNS = [
    r"\binfant\b.*\bfever\b",
    r"\bhigh\s*fever\b.*\b(baby|infant|child)\b",
    r"\b(?:bache|bachhe|shishu)\s+ko\s+tez\s+bukhar\b",
    r"(?:बच्चे|शिशु)\s+को\s+तेज\s+बुखार",
    r"\bfracture\b",
    r"\bbroken\s*(bone|arm|leg|wrist|ankle)\b",
    r"\b(?:haddi\s+toot\s+gayi|haddi\s+tooti)\b",
    r"हड्डी\s+टूट",
    r"\bfall\b.*\b(pain|swelling|fracture)\b",
    r"\b(?:gir\s+gaya|gir\s+gaye).*(?:dard|soojan|fracture)\b",
    r"\bvomiting\b.*(stop|non\s*-?stop|persist|blood)",
    r"\b(?:ulti\s+nahi\s+ruk\s+rahi)\b",
    r"उल्टी\s+नहीं\s+रुक\s+रही",
    r"\bdeep\s*cut\b",
    r"\bstitches\b",
    r"\b(?:gehra\s+ghav|gehra\s+cut|taanke)\b",
    r"(?:गहरा\s+घाव|टांके)",
]

LEVEL_3_PATTERNS = [
    r"\bnurse\s*(to\s*)?(look|check|call|visit)\b",
    r"\b(?:nurse\s+ko\s+bhejo|nurse\s+chahiye|nurse\s+visit)\b",
    r"नर्स\s+(?:को\s+भेजो|चाहिए|विजिट)",
    r"\bmild\s*fever\b",
    r"\b(?:halka\s+bukhar)\b",
    r"हल्का\s+बुखार",
    r"\brash\b",
    r"\bdizziness\b",
    r"\b(?:chakkar\s+aa\s+rahe|chakkar\s+aa\s+raha)\b",
    r"चक्कर",
    r"\broutine\s*check\b",
]


def assess_safety(message: str, classifier_fn=None) -> SafetyAssessment:
    text = (message or "").strip()
    lowered = text.lower()

    if classifier_fn is not None:
        try:
            res = classifier_fn(text)
            if isinstance(res, SafetyAssessment):
                return res
            if isinstance(res, dict) and res.get("level") in (1, 2, 3):
                lvl = int(res["level"])
                sla = {1: 5, 2: 30, 3: 240}[lvl]
                return SafetyAssessment(
                    level=lvl,
                    reason=str(res.get("reason", "Classifier escalation")),
                    call_112=(lvl == 1),
                    sla_minutes=sla,
                )
        except Exception:
            return SafetyAssessment(
                level=1,
                reason="Safety classifier error/timeout; defaulting to Level 1 emergency escalation.",
                call_112=True,
                sla_minutes=5,
            )

    for pat in LEVEL_1_PATTERNS:
        if re.search(pat, lowered) or re.search(pat, text):
            return SafetyAssessment(level=1, reason=f"Matched critical symptom pattern: {pat}", call_112=True, sla_minutes=5)

    for pat in LEVEL_2_PATTERNS:
        if re.search(pat, lowered) or re.search(pat, text):
            return SafetyAssessment(level=2, reason=f"Matched urgent symptom pattern: {pat}", call_112=False, sla_minutes=30)

    for pat in LEVEL_3_PATTERNS:
        if re.search(pat, lowered) or re.search(pat, text):
            return SafetyAssessment(level=3, reason=f"Matched routine nurse review pattern: {pat}", call_112=False, sla_minutes=240)

    return SafetyAssessment(level=None, reason="No emergency or nurse escalation indicators detected.", call_112=False, sla_minutes=None)
