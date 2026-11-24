import asyncio
import re
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.classifier import Intent, SafetyAndIntentClassifier, SafetyVerdict, Service
from app.models import Patient, Society
from app.rag import ABSTENTION_MESSAGE, retrieve_corpus_chunks
from app.tools import (
    CONTRACT_TOOLS,
    BookServiceArgs,
    ListServiceRequestsArgs,
    LogSymptomArgs,
    SearchClinicDocsArgs,
    CareConciergeToolbox,
    _close_session,
    _open_session,
)

MAX_TOOL_CALLS_PER_TURN = 4
DEFAULT_STEP_TIMEOUT_SECONDS = 5.0


class ConciergeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    patient_id: str
    society_id: str
    request_id: str
    intent: str
    safety_verdict: str
    safety_level: int | None
    safety_failed: bool
    safety_fallback_triggered: bool
    tool_calls_count: int
    pending_booking: dict[str, Any] | None
    pending_symptom: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    reply: str
    escalation_id: str | None
    error: str | None


def _extract_last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        role = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return str(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else ""))
    if messages:
        last = messages[-1]
        return str(getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else last))
    return ""


def _parse_iso_slot(text: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)", text or "")
    if not m:
        return None
    date_part = m.group(1)
    time_part = m.group(2)
    if len(time_part) == 5:
        time_part = f"{time_part}:00"
    return f"{date_part}T{time_part}+05:30"


def _detect_language(text: str) -> str:
    if any("\u0900" <= ch <= "\u097f" for ch in (text or "")):
        return "hi"
    hinglish_markers = {"kya", "kab", "hai", "hain", "mera", "meri", "book", "karna", "chahiye", "bukhar", "dard"}
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if len(tokens & hinglish_markers) >= 2:
        return "hinglish"
    return "en"


def _extract_symptom_data(text: str) -> dict[str, Any]:
    lowered = (text or "").lower()
    sev = 3
    sev_match = re.search(r"\b(?:severity|level|scale|rating)?\s*([1-5])\b", lowered)
    if sev_match:
        try:
            sev = int(sev_match.group(1))
        except ValueError:
            pass
    elif any(w in lowered for w in ("mild", "halka")):
        sev = 2
    elif any(w in lowered for w in ("moderate", "theek")):
        sev = 3
    elif any(w in lowered for w in ("severe", "tez", "bad")):
        sev = 4

    cleaned = re.sub(r"(?i)\b(log|record|my|symptom|severity|level|is|rating|scale|\d)\b", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :,-")
    symptom = cleaned if cleaned else text
    return {"symptom": symptom, "severity": sev}


def _user_supplied_severity(text: str) -> bool:
    return bool(
        re.search(r"\b(?:severity|level|scale|rating)?\s*[1-5]\b", text or "", re.I)
        or re.search(r"\b(?:mild|moderate|severe|halka|tez)\b", text or "", re.I)
    )


def _user_supports_booking_choice(
    *, user_message: str, pending: dict[str, Any] | None, service: str, subtype: str
) -> bool:
    """Ensure an LLM tool call cannot supply business facts absent from the turn."""
    lower = (user_message or "").lower()
    if pending and str(pending.get("service", "")).upper() == service and str(
        pending.get("subtype", "")
    ).upper() == subtype:
        return True
    if service == Service.LAB.value:
        lab_markers = {
            "CBC": ("cbc", "complete blood count"),
            "FASTING_BLOOD_SUGAR": ("fasting blood sugar", "fbs"),
            "HBA1C": ("hba1c",),
            "LIPID_PROFILE": ("lipid",),
            "LFT": ("lft", "liver function"),
            "KFT": ("kft", "kidney function"),
            "THYROID_PROFILE": ("thyroid",),
            "VITAMIN_D": ("vitamin d",),
            "URINE_ROUTINE": ("urine",),
        }
        return any(marker in lower for marker in lab_markers.get(subtype, (subtype.lower(),)))
    subtype_markers = {
        "GENERAL_PHYSICIAN": ("general physician", "doctor", "consult"),
        "CARDIOLOGY": ("cardio",),
        "PAEDIATRICS": ("paediatric", "pediatric", "child"),
        "ORTHOPAEDICS": ("ortho",),
        "DERMATOLOGY": ("derma", "skin"),
        "ENT": ("ent",),
        "GYNAECOLOGY": ("gyn",),
        "PHYSIOTHERAPY": ("physio",),
    }
    return any(marker in lower for marker in subtype_markers.get(subtype, (subtype.lower(),)))


def _route_bound_tool_call(
    tool_call: dict[str, Any], *, state: ConciergeState, user_message: str, pending: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate a LangChain tool call and translate it to trusted graph state.

    Model-provided patient, society, and request IDs are discarded and replaced
    with the authenticated state values. A model cannot turn missing user facts
    into a database write: slot, service, subtype, and symptom severity are
    cross-checked against this turn or a persisted pending booking.
    """
    name = tool_call.get("name")
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return {"error": f"Malformed tool arguments for {name or 'unknown tool'}."}

    patient_id = str(state.get("patient_id") or "")
    request_id = str(state.get("request_id") or "")
    society_id = str(state.get("society_id") or "")
    try:
        if name == "book_service":
            parsed = BookServiceArgs.model_validate({**args, "patient_id": patient_id, "request_id": request_id})
            service = parsed.service.value
            subtype = str(parsed.subtype).upper()
            trusted_slot = _parse_iso_slot(user_message) or (
                pending.get("scheduled_for") if isinstance(pending, dict) else None
            )
            supplied_slot = _parse_iso_slot(parsed.scheduled_for)
            if not trusted_slot:
                return {
                    "intent": Intent.BOOK_SERVICE.value,
                    "pending_booking": {"service": service, "subtype": subtype, "scheduled_for": None},
                }
            if supplied_slot != trusted_slot or not _user_supports_booking_choice(
                user_message=user_message, pending=pending, service=service, subtype=subtype
            ):
                return {"error": "Booking tool arguments include details not supplied by the patient."}
            return {
                "intent": Intent.BOOK_SERVICE.value,
                "pending_booking": {"service": service, "subtype": subtype, "scheduled_for": trusted_slot},
            }
        if name == "list_service_requests":
            ListServiceRequestsArgs.model_validate({**args, "patient_id": patient_id})
            return {"intent": Intent.LIST_SERVICES.value}
        if name == "log_symptom":
            parsed = LogSymptomArgs.model_validate({**args, "patient_id": patient_id, "request_id": request_id})
            if not _user_supplied_severity(user_message):
                return {"error": "Please provide a symptom severity from 1 to 5."}
            return {
                "intent": Intent.LOG_SYMPTOM.value,
                "pending_symptom": {"symptom": parsed.symptom, "severity": parsed.severity},
            }
        if name == "search_clinic_docs":
            SearchClinicDocsArgs.model_validate({**args, "query": user_message, "society_id": society_id})
            return {"intent": Intent.CLINIC_INFO.value}
        # Emergency escalation is deliberately unavailable to the SAFE router;
        # only the safety node may invoke it.
        if name == "escalate_emergency":
            return {"error": "Emergency escalation is only available through the safety gate."}
    except Exception as exc:
        return {"error": f"Malformed tool arguments for {name or 'unknown tool'}: {exc}"}
    return {"error": f"Unsupported tool call: {name or 'unknown tool'}."}


def _append_citation_labels(reply: str, citations: list[dict[str, Any]]) -> str:
    """Keep source attribution in the patient-visible final answer and SSE data."""
    labels = []
    for citation in citations:
        label = citation.get("heading_path") or citation.get("section") or citation.get("title")
        if label and label not in labels:
            labels.append(str(label))
    return f"{reply.rstrip()}\n\nSources: {'; '.join(labels)}" if labels else reply


def build_care_concierge_graph(
    session_factory=None,
    classifier: SafetyAndIntentClassifier | None = None,
    classifier_fn=None,
    toolbox: CareConciergeToolbox | None = None,
    checkpointer=None,
    llm=None,
    step_timeout: float = DEFAULT_STEP_TIMEOUT_SECONDS,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
):
    """Build the compiled LangGraph StateGraph for MicroHeal Care Concierge."""
    active_llm = llm
    active_classifier = classifier or SafetyAndIntentClassifier(
        llm=active_llm,
        classifier_fn=classifier_fn,
        timeout_seconds=step_timeout,
    )
    active_toolbox = toolbox or CareConciergeToolbox(session_factory=session_factory)
    # Bind Pydantic-backed contract tools once per graph. Identity fields still
    # come from trusted graph state, never from model-generated arguments.
    router_llm = active_llm
    if active_llm is not None and hasattr(active_llm, "bind_tools"):
        try:
            router_llm = active_llm.bind_tools(CONTRACT_TOOLS)
        except Exception:
            router_llm = active_llm

    async def safety_node(state: ConciergeState) -> dict[str, Any]:
        user_msg = _extract_last_user_text(state.get("messages") or [])
        req_id = state.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
        patient_id = state.get("patient_id") or "pat_001"
        thread_id = state.get("thread_id") or "default_thread"

        soc_id = state.get("society_id") or ""
        if not soc_id and (session_factory or active_toolbox.session_factory or active_toolbox.session):
            try:
                db, cleanup = await _open_session(
                    session_factory or active_toolbox.session_factory or active_toolbox.session
                )
                try:
                    patient = await asyncio.wait_for(db.get(Patient, patient_id), timeout=step_timeout)
                    if patient:
                        soc_id = str(patient.society_id)
                finally:
                    await _close_session(cleanup)
            except Exception:
                pass

        prev_failures = int(getattr(active_classifier, "failure_count", 0))
        try:
            cls_out = await asyncio.wait_for(
                active_classifier.classify(user_msg, context={"patient_id": patient_id, "society_id": soc_id}),
                timeout=step_timeout,
            )
            verdict = str(cls_out.get("verdict", SafetyVerdict.SAFE.value))
            intent = str(cls_out.get("intent", Intent.UNKNOWN.value))
            raw_level = cls_out.get("safety_level")
            level = int(raw_level) if raw_level is not None else None
            failed = bool(cls_out.get("classifier_failed", False))
        except Exception:
            if int(getattr(active_classifier, "failure_count", 0)) == prev_failures:
                active_classifier.failure_count = prev_failures + 1
            verdict = SafetyVerdict.EMERGENCY.value
            intent = Intent.EMERGENCY.value
            level = 1
            failed = True

        if failed or verdict == SafetyVerdict.EMERGENCY.value:
            esc_level = int(level or 1)
            sla = 5 if esc_level == 1 else (30 if esc_level == 2 else 240)
            calls: list[dict[str, Any]] = []
            esc_id: str | None = None
            err_msg: str | None = None

            if not failed and len(calls) < max_tool_calls:
                try:
                    esc_res = await asyncio.wait_for(
                        active_toolbox.escalate_emergency(
                            patient_id=patient_id,
                            transcript=user_msg,
                            request_id=req_id,
                            thread_id=thread_id,
                            level=esc_level,
                            sla_minutes=sla,
                        ),
                        timeout=step_timeout,
                    )
                    esc_id = str(esc_res.get("escalation_id")) if esc_res.get("escalation_id") else None
                    calls.append({"tool": "escalate_emergency", "result": esc_res})
                except Exception as exc:
                    err_msg = str(exc)

            lang = _detect_language(user_msg)
            if lang == "hinglish":
                reply = (
                    f"Emergency (Level {esc_level}, {esc_id or 'ESCALATED'}): Turant **112** par call karein "
                    f"aur callback ka intezaar na karein. Escalation store ho gaya hai (SLA: {sla} minutes). "
                    f"Is demo me nurse ko automatic paging nahi hai, isliye kisi nurse se contact nahi kiya gaya."
                )
            elif lang == "hi":
                reply = (
                    f"आपातकालीन स्थिति (स्तर {esc_level}, {esc_id or 'ESCALATED'}): कृपया तुरंत **112** पर कॉल करें "
                    f"और कॉलबैक की प्रतीक्षा न करें। एस्केलेशन दर्ज कर लिया गया है (SLA: {sla} मिनट)। "
                    f"इस डेमो में नर्स को स्वचालित पेजिंग कॉन्फ़िगर नहीं है, इसलिए किसी नर्स को स्वतः सूचित नहीं किया गया है।"
                )
            else:
                reply = (
                    f"Emergency Escalation (Level {esc_level}, {esc_id or 'LOGGED'}): Please call **112** "
                    f"immediately and do not wait for a callback. The escalation has been stored "
                    f"(SLA: {sla} minutes). This demo has no external nurse paging configured, so no nurse "
                    f"has been contacted automatically."
                )

            return {
                "society_id": soc_id,
                "safety_verdict": SafetyVerdict.EMERGENCY.value,
                "safety_level": esc_level,
                "safety_failed": bool(failed),
                "safety_fallback_triggered": bool(failed),
                "tool_calls_count": len(calls),
                "intent": Intent.EMERGENCY.value,
                "escalation_id": esc_id,
                "tool_calls": calls,
                "reply": reply,
                "error": err_msg,
                "messages": [AIMessage(content=reply)],
            }

        return {
            "society_id": soc_id,
            "safety_verdict": verdict,
            "safety_level": level,
            "safety_failed": False,
            "safety_fallback_triggered": False,
            "tool_calls_count": 0,
            "intent": intent,
            "pending_symptom": None,
            "tool_calls": [],
            "citations": [],
            "escalation_id": None,
            "error": None,
        }

    def route_after_safety(state: ConciergeState) -> str:
        if state.get("safety_verdict") == SafetyVerdict.EMERGENCY.value or state.get("safety_failed"):
            return "end"
        if state.get("safety_verdict") == SafetyVerdict.MEDICAL_ADVICE.value:
            return "response"
        return "router"

    async def router_node(state: ConciergeState) -> dict[str, Any]:
        user_msg = _extract_last_user_text(state.get("messages") or [])
        lower = user_msg.lower()
        intent = state.get("intent") or Intent.UNKNOWN.value
        pending = state.get("pending_booking")
        parsed_slot = _parse_iso_slot(user_msg)

        # 1. LLM-assisted routing uses LangChain's structured tool-call
        # protocol. Free-form JSON is intentionally not accepted here.
        if router_llm is not None:
            try:
                system_prompt = (
                    "You are the clinical concierge router. Use exactly one bound tool only when the patient "
                    "has supplied its required business details. For identity fields, use only these runtime "
                    f"values: patient_id={state.get('patient_id')}, society_id={state.get('society_id')}, "
                    f"request_id={state.get('request_id')}. Never invent a date, time, specialty, symptom, "
                    "or severity. If details are missing, do not call a tool; answer with no tool call."
                )
                llm_res = await asyncio.wait_for(
                    router_llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]),
                    timeout=min(step_timeout, 4.0),
                )
                tool_calls = list(getattr(llm_res, "tool_calls", None) or [])
                if tool_calls:
                    # One action per turn keeps side effects bounded and makes
                    # the graph-level cap auditable.
                    return _route_bound_tool_call(
                        tool_calls[0], state=state, user_message=user_msg, pending=pending
                    )
            except Exception:
                # A router-provider failure falls back to deterministic intent
                # extraction; it never bypasses the preceding safety gate.
                pass

        # 2. Deterministic Fallback Routing
        if intent == Intent.LIST_SERVICES.value or any(
            k in lower for k in ("upcoming", "appointments", "past bookings", "my bookings", "list bookings")
        ):
            return {"intent": Intent.LIST_SERVICES.value}

        if intent == Intent.LOG_SYMPTOM.value or any(
            k in lower for k in ("log symptom", "symptom", "feeling feverish", "bukhar", "sir dard", "headache", "chest pain", "pain")
        ) and any(k in lower for k in ("log", "severity", "level", "1", "2", "3", "4", "5", "record", "feeling", "mild", "moderate")):
            sym_data = _extract_symptom_data(user_msg)
            return {
                "intent": Intent.LOG_SYMPTOM.value,
                "pending_symptom": sym_data,
            }

        if (
            intent == Intent.BOOK_SERVICE.value
            or "book" in lower
            or "cardiology" in lower
            or (parsed_slot is not None and pending is not None)
            or parsed_slot is not None
        ):
            subtype = None
            if "cardio" in lower:
                subtype = "CARDIOLOGY"
            elif any(w in lower for w in ("general physician", "doctor", "consult")):
                subtype = "GENERAL_PHYSICIAN"
            elif "lipid" in lower:
                subtype = "LIPID_PROFILE"
            elif "thyroid" in lower:
                subtype = "THYROID_PROFILE"
            elif "cbc" in lower:
                subtype = "CBC"
            elif isinstance(pending, dict) and pending.get("subtype"):
                subtype = str(pending["subtype"])

            service = None
            if any(w in lower for w in ("lab", "blood", "lipid", "thyroid", "fasting", "cbc")):
                service = Service.LAB.value
            elif subtype in ("CARDIOLOGY", "GENERAL_PHYSICIAN"):
                service = Service.DOCTOR.value
            elif isinstance(pending, dict) and pending.get("service"):
                service = str(pending["service"])
            slot_val = parsed_slot or (pending.get("scheduled_for") if isinstance(pending, dict) else None)
            return {
                "intent": Intent.BOOK_SERVICE.value,
                "pending_booking": {
                    "service": service,
                    "subtype": subtype,
                    "scheduled_for": slot_val,
                },
            }

        return {"intent": Intent.CLINIC_INFO.value if intent == Intent.UNKNOWN.value else intent}

    async def tool_node(state: ConciergeState) -> dict[str, Any]:
        calls = list(state.get("tool_calls") or [])
        if len(calls) >= max_tool_calls:
            return {
                "tool_calls": calls,
                "tool_calls_count": len(calls),
                "error": "Tool call limit reached for turn.",
            }

        intent = state.get("intent") or Intent.CLINIC_INFO.value
        patient_id = state.get("patient_id") or "pat_001"
        req_id = state.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
        thread_id = state.get("thread_id") or "default_thread"
        user_msg = _extract_last_user_text(state.get("messages") or [])
        soc_id = state.get("society_id") or "soc_shivalik"
        citations: list[dict[str, Any]] = []
        err_out: str | None = None

        # Symptom Logging Tool Call
        if intent == Intent.LOG_SYMPTOM.value or state.get("pending_symptom"):
            sym_data = state.get("pending_symptom") or _extract_symptom_data(user_msg)
            try:
                res = await asyncio.wait_for(
                    active_toolbox.log_symptom(
                        patient_id=patient_id,
                        symptom=str(sym_data.get("symptom") or "Reported symptom"),
                        severity=int(sym_data.get("severity") or 3),
                        request_id=req_id,
                        thread_id=thread_id,
                    ),
                    timeout=step_timeout,
                )
                calls.append({"tool": "log_symptom", "result": res})
                return {
                    "tool_calls": calls,
                    "tool_calls_count": len(calls),
                    "pending_symptom": None,
                    "error": None,
                }
            except Exception as exc:
                err_out = str(exc)
                calls.append({"tool": "log_symptom", "error": err_out})
                return {
                    "tool_calls": calls,
                    "tool_calls_count": len(calls),
                    "error": err_out,
                }

        # Booking Tool Call
        if intent == Intent.BOOK_SERVICE.value:
            pending = state.get("pending_booking") or {}
            sched = pending.get("scheduled_for")
            if sched and pending.get("service") and pending.get("subtype"):
                try:
                    res = await asyncio.wait_for(
                        active_toolbox.book_service(
                            patient_id=patient_id,
                            service=str(pending.get("service", Service.DOCTOR.value)),
                            subtype=str(pending.get("subtype", "GENERAL_PHYSICIAN")),
                            scheduled_for=str(sched),
                            request_id=req_id,
                            thread_id=thread_id,
                        ),
                        timeout=step_timeout,
                    )
                    calls.append({"tool": "book_service", "result": res})
                    return {
                        "tool_calls": calls,
                        "tool_calls_count": len(calls),
                        "pending_booking": None,
                        "error": None,
                    }
                except Exception as exc:
                    err_out = str(exc)
                    calls.append({"tool": "book_service", "error": err_out})
                    return {
                        "tool_calls": calls,
                        "tool_calls_count": len(calls),
                        "error": err_out,
                    }
            return {"tool_calls": calls, "tool_calls_count": len(calls)}

        # List Services Tool Call
        if intent == Intent.LIST_SERVICES.value:
            try:
                reqs = await asyncio.wait_for(
                    active_toolbox.list_service_requests(patient_id=patient_id),
                    timeout=step_timeout,
                )
                calls.append({"tool": "list_service_requests", "result": reqs})
                return {"tool_calls": calls, "tool_calls_count": len(calls), "error": None}
            except Exception as exc:
                err_out = str(exc)
                calls.append({"tool": "list_service_requests", "error": err_out})
                return {"tool_calls": calls, "tool_calls_count": len(calls), "error": err_out}

        # Clinic Docs Search (RAG) Tool Call
        try:
            db, cleanup = await _open_session(
                session_factory or active_toolbox.session_factory or active_toolbox.session
            )
            try:
                society = await asyncio.wait_for(db.get(Society, soc_id), timeout=step_timeout)
                rag_res = await asyncio.wait_for(
                    retrieve_corpus_chunks(
                        db,
                        user_msg,
                        society=society,
                        society_id=soc_id,
                        top_k=6,
                        include_internal=False,
                    ),
                    timeout=step_timeout,
                )
                calls.append({"tool": "search_clinic_docs", "result": rag_res})
                citations = list(rag_res.get("citations") or [])
            finally:
                await _close_session(cleanup)
        except Exception as exc:
            err_out = str(exc)
            calls.append({"tool": "search_clinic_docs", "error": err_out})

        return {
            "tool_calls": calls,
            "tool_calls_count": len(calls),
            "citations": citations,
            "error": err_out,
        }

    async def response_node(state: ConciergeState) -> dict[str, Any]:
        verdict = state.get("safety_verdict") or SafetyVerdict.SAFE.value
        user_msg = _extract_last_user_text(state.get("messages") or [])
        lang = _detect_language(user_msg)
        calls = state.get("tool_calls") or []
        citations = list(state.get("citations") or [])

        if verdict == SafetyVerdict.MEDICAL_ADVICE.value:
            if lang in ("hi", "hinglish"):
                reply = (
                    "मैं चिकित्सा निदान, दवा या खुराक की सलाह नहीं दे सकता। "
                    "कृपया इसके लिए डॉक्टर परामर्श बुक करें।"
                )
            else:
                reply = (
                    "I cannot provide a medical diagnosis, prescription, or medication dosage. "
                    "Please book a doctor consultation for medical advice."
                )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        if state.get("error"):
            reply = (
                f"अस्थायी सिस्टम त्रुटि हुई ({state['error']})। कृपया फिर कोशिश करें।"
                if lang in ("hi", "hinglish")
                else f"I encountered a temporary system error ({state['error']}). Please try again."
            )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        # Symptom Logging response
        sym_calls = [c for c in calls if c.get("tool") == "log_symptom" and "result" in c]
        if sym_calls:
            s_res = sym_calls[-1]["result"]
            reply = (
                f"आपके केयर लॉग में दर्ज किया गया: '{s_res.get('symptom')}' (गंभीरता: {s_res.get('severity')}/5)। "
                f"लॉग ID: {s_res.get('id')}. लक्षण बने रहें या बढ़ें तो क्लिनिक डॉक्टर से संपर्क करें।"
                if lang in ("hi", "hinglish") else
                f"Recorded in your care log: '{s_res.get('symptom')}' (Severity: {s_res.get('severity')}/5). "
                f"Log ID: {s_res.get('id')}. If your symptoms persist or worsen, please consult our clinic doctor."
            )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        intent = state.get("intent") or Intent.CLINIC_INFO.value
        if intent == Intent.BOOK_SERVICE.value:
            book_calls = [c for c in calls if c.get("tool") == "book_service" and "result" in c]
            if book_calls:
                res = book_calls[-1]["result"]
                subtype_val = str(res.get("subtype") or "CONSULTATION").upper()
                reply = (
                    f"बुकिंग पुष्टि हुई [{subtype_val}] — बुकिंग ID: {res.get('booking_id') or res.get('id')}, "
                    f"समय: {res.get('scheduled_for')}।"
                    if lang in ("hi", "hinglish") else
                    f"Booking confirmed [{subtype_val}] — Booking ID: {res.get('booking_id') or res.get('id')}, "
                    f"Scheduled for: {res.get('scheduled_for')}."
                )
            else:
                pending = state.get("pending_booking") or {}
                missing: list[str] = []
                if not pending.get("subtype"):
                    missing.append("specialty or lab test")
                if not pending.get("scheduled_for"):
                    missing.append("preferred date and time")
                detail = " and ".join(missing) or "preferred date and time"
                sub_hint = str(pending.get("subtype") or "appointment").lower()
                reply = (
                    f"मैं आपका {sub_hint} अपॉइंटमेंट बुक कर सकता हूँ। कृपया {detail} दें "
                    "(उदाहरण: 2027-03-02, 10:00)।"
                    if lang in ("hi", "hinglish") else
                    f"I can help book a {sub_hint} appointment for you. Please share your {detail} "
                    "(for example, 2027-03-02 at 10:00)."
                )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        if intent == Intent.LIST_SERVICES.value:
            list_calls = [c for c in calls if c.get("tool") == "list_service_requests" and "result" in c]
            rows = list_calls[-1]["result"] if list_calls else []
            reply = (
                f"आपकी {len(rows)} सेवा रिक्वेस्ट रिकॉर्ड में है।"
                if lang in ("hi", "hinglish") else
                f"You have {len(rows)} service request(s) on file."
            )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        rag_calls = [c for c in calls if c.get("tool") == "search_clinic_docs" and "result" in c]
        rag_res = rag_calls[-1]["result"] if rag_calls else {}
        chunks = rag_res.get("chunks") or []
        if rag_res.get("abstained") or not chunks:
            reply = (
                "मौजूदा क्लिनिक नीति और आपकी सोसाइटी एनेक्स के आधार पर मुझे इसका उत्तर नहीं पता। "
                "क्या आप नर्स से सहायता लेना चाहेंगे?"
                if lang in ("hi", "hinglish") else ABSTENTION_MESSAGE
            )
            return {"reply": reply, "citations": [], "messages": [AIMessage(content=reply)]}

        # If LLM is available, generate natural response from retrieved chunks
        if active_llm is not None:
            try:
                context_str = "\n\n".join(f"[{c['heading_path']}]\n{c['content']}" for c in chunks[:3])
                sys_msg = (
                    "You are a helpful healthcare concierge assistant. Answer the user question accurately and concisely "
                    f"using only the following clinic context. Reply in the user's language ({lang}); "
                    "if you don't know, state so.\n\nContext:\n" + context_str
                )
                llm_ans = await asyncio.wait_for(
                    active_llm.ainvoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]),
                    timeout=min(step_timeout, 4.0),
                )
                ans_text = getattr(llm_ans, "content", "").strip()
                if ans_text:
                    reply = _append_citation_labels(ans_text, citations)
                    return {"reply": reply, "citations": citations, "messages": [AIMessage(content=reply)]}
            except Exception:
                pass

        top = chunks[0]
        reply = _append_citation_labels(str(top["content"]), citations)
        return {"reply": reply, "citations": citations, "messages": [AIMessage(content=reply)]}

    builder = StateGraph(ConciergeState)
    builder.add_node("safety", safety_node)
    builder.add_node("router", router_node)
    builder.add_node("tools", tool_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "safety")
    builder.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "end": END,
            "router": "router",
            "response": "response",
        },
    )
    builder.add_edge("router", "tools")
    builder.add_edge("tools", "response")
    builder.add_edge("response", END)

    return builder.compile(checkpointer=checkpointer)
