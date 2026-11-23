import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import AsyncSessionLocal, check_db_connection, get_db
from app.graph import build_care_concierge_graph
from app.models import Escalation, Patient, Society
from app.tools import CareConciergeToolbox

settings = get_settings()

_CHECKPOINTER_POOL: AsyncConnectionPool | None = None
_CHECKPOINTER: AsyncPostgresSaver | None = None
_COMPILED_GRAPH: Any = None


def _sync_pg_conninfo() -> str:
    raw = os.getenv("SYNC_DATABASE_URL") or settings.sync_database_url or settings.database_url
    return (
        raw.replace("postgresql+psycopg://", "postgresql://", 1)
        .replace("postgresql+asyncpg://", "postgresql://", 1)
    )


async def _get_or_init_graph():
    global _CHECKPOINTER_POOL, _CHECKPOINTER, _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    try:
        conninfo = _sync_pg_conninfo()
        pool = AsyncConnectionPool(
            conninfo=conninfo,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        _CHECKPOINTER_POOL = pool
        _CHECKPOINTER = saver
    except Exception:
        _CHECKPOINTER = None

    llm = None
    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm_kwargs = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": settings.openai_api_key,
                "temperature": 0.0,
            }
            if settings.openai_base_url:
                llm_kwargs["base_url"] = settings.openai_base_url
            llm = ChatOpenAI(
                **llm_kwargs,
            )
        except Exception:
            llm = None

    toolbox = CareConciergeToolbox(session_factory=AsyncSessionLocal)
    _COMPILED_GRAPH = build_care_concierge_graph(
        session_factory=AsyncSessionLocal,
        toolbox=toolbox,
        checkpointer=_CHECKPOINTER,
        llm=llm,
    )
    return _COMPILED_GRAPH


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _get_or_init_graph()
    try:
        yield
    finally:
        global _CHECKPOINTER_POOL, _CHECKPOINTER, _COMPILED_GRAPH
        if _CHECKPOINTER_POOL is not None:
            try:
                await _CHECKPOINTER_POOL.close()
            except Exception:
                pass
        _CHECKPOINTER_POOL = None
        _CHECKPOINTER = None
        _COMPILED_GRAPH = None


app = FastAPI(title="MicroHeal Care Concierge", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    # The client owns this UUID and must resend it unchanged on a retry.
    # Generating it server-side would make a lost response non-idempotent.
    request_id: str = Field(..., min_length=1)
    language: str | None = None
    book_slot: dict[str, str] | None = None


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _token_chunks(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(\s+)", text)
    return [p for p in parts if p]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    db_ok = await check_db_connection()
    if not db_ok:
        raise HTTPException(status_code=503, detail={"status": "error", "db": "disconnected"})
    return {"status": "ok", "db": "connected"}


@app.post("/v1/chat")
async def chat_sse_endpoint(payload: ChatRequest, request: Request) -> StreamingResponse:
    req_id = payload.request_id
    graph = await _get_or_init_graph()

    async def event_stream():
        try:
            async with AsyncSessionLocal() as db:
                patient = await db.get(Patient, payload.patient_id)
                if not patient:
                    yield _sse_event(
                        "error",
                        {
                            "thread_id": payload.thread_id,
                            "request_id": req_id,
                            "message": f"Unknown patient_id: {payload.patient_id}",
                        },
                    )
                    return
                society_id = patient.society_id

            config = {"configurable": {"thread_id": payload.thread_id}}
            state_input: dict[str, Any] = {
                "messages": [HumanMessage(content=payload.message)],
                "thread_id": payload.thread_id,
                "patient_id": payload.patient_id,
                "society_id": society_id,
                "request_id": req_id,
            }
            if payload.book_slot:
                state_input["pending_booking"] = payload.book_slot

            result = await graph.ainvoke(state_input, config=config)

            if result.get("error"):
                yield _sse_event(
                    "error",
                    {
                        "thread_id": payload.thread_id,
                        "request_id": req_id,
                        "message": result["error"],
                    },
                )
                return

            for tc in (result.get("tool_calls") or []):
                tool_name = tc.get("tool", "tool")
                yield _sse_event(
                    "tool_call",
                    {
                        "thread_id": payload.thread_id,
                        "request_id": req_id,
                        "tool": tool_name,
                        "args": tc.get("args", {}),
                    },
                )
                yield _sse_event(
                    "tool_result",
                    {
                        "thread_id": payload.thread_id,
                        "request_id": req_id,
                        "tool": tool_name,
                        "result": tc.get("result", tc.get("error")),
                    },
                )

            for cit in (result.get("citations") or []):
                yield _sse_event(
                    "citation",
                    {
                        "thread_id": payload.thread_id,
                        "request_id": req_id,
                        "citation": cit,
                    },
                )

            reply_text = result.get("reply") or ""
            for tok in _token_chunks(reply_text):
                yield _sse_event(
                    "token",
                    {
                        "thread_id": payload.thread_id,
                        "request_id": req_id,
                        "token": tok,
                    },
                )
                await asyncio.sleep(0.005)

            booking_obj = None
            for tc in reversed(result.get("tool_calls") or []):
                if tc.get("tool") == "book_service" and isinstance(tc.get("result"), dict):
                    booking_obj = tc["result"]
                    break

            yield _sse_event(
                "final",
                {
                    "thread_id": payload.thread_id,
                    "request_id": req_id,
                    "reply": reply_text,
                    "intent": result.get("intent"),
                    "safety_verdict": result.get("safety_verdict"),
                    "escalation_id": result.get("escalation_id"),
                    "booking": booking_obj,
                    "citations": result.get("citations") or [],
                    "tool_calls": result.get("tool_calls") or [],
                },
            )
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "thread_id": payload.thread_id,
                    "request_id": req_id,
                    "message": str(exc),
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/threads/{thread_id}")
async def get_thread_transcript(thread_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Read thread transcript directly from the LangGraph Postgres checkpointer."""
    graph = await _get_or_init_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state_values: dict[str, Any] = {}
    if getattr(graph, "checkpointer", None) is not None and hasattr(graph, "aget_state"):
        try:
            snap = await graph.aget_state(config)
            if snap is not None and getattr(snap, "values", None):
                state_values = dict(snap.values)
        except Exception:
            state_values = {}

    raw_messages = state_values.get("messages") or []
    messages_out: list[dict[str, Any]] = []
    for idx, msg in enumerate(raw_messages):
        if isinstance(msg, HumanMessage):
            role = "user"
            content = str(msg.content)
        elif isinstance(msg, AIMessage):
            role = "assistant"
            content = str(msg.content)
        else:
            msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "assistant")
            role = "user" if msg_type in ("human", "user") else "assistant"
            content = str(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else msg))
        messages_out.append(
            {
                "index": idx,
                "role": role,
                "content": content,
            }
        )

    escalations = (
        await db.execute(select(Escalation).where(Escalation.thread_id == thread_id))
    ).scalars().all()

    if not messages_out and not state_values and not escalations:
        raise HTTPException(status_code=404, detail="Thread not found")

    return {
        "thread_id": thread_id,
        "patient_id": state_values.get("patient_id", ""),
        "society_id": state_values.get("society_id", ""),
        "intent": state_values.get("intent"),
        "safety_verdict": state_values.get("safety_verdict"),
        "escalation_id": state_values.get("escalation_id"),
        "tool_calls": state_values.get("tool_calls") or [],
        "citations": state_values.get("citations") or [],
        "messages": messages_out,
        "escalations": [
            {
                "id": e.id,
                "request_id": e.request_id,
                "level": e.level,
                "sla_minutes": e.sla_minutes,
                "status": e.status,
                "trigger_message": e.trigger_message,
            }
            for e in escalations
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def ui_home() -> HTMLResponse:
    static_html = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if static_html.exists():
        return HTMLResponse(content=static_html.read_text(encoding="utf-8"))
    app_static_html = Path(__file__).resolve().parent / "static" / "index.html"
    if app_static_html.exists():
        return HTMLResponse(content=app_static_html.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>MicroHeal Care Concierge</h1>")
