import os
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.graph import build_care_concierge_graph
from app.tools import CareConciergeToolbox


def _pg_conninfo() -> str:
    settings = get_settings()
    raw = os.getenv("SYNC_DATABASE_URL") or settings.sync_database_url or settings.database_url
    return (
        raw.replace("postgresql+psycopg://", "postgresql://", 1)
        .replace("postgresql+asyncpg://", "postgresql://", 1)
    )


async def run_concierge_turn(
    session_factory: Any,
    *,
    thread_id: str,
    patient_id: str,
    message: str,
    request_id: str | None = None,
    classifier_fn: Any = None,
    checkpointer: Any = None,
) -> dict[str, Any]:
    """Execute a single concierge turn using the compiled LangGraph StateGraph."""
    toolbox = CareConciergeToolbox(session_factory=session_factory)
    pool: AsyncConnectionPool | None = None
    active_cp = checkpointer
    try:
        if active_cp is None:
            pool = AsyncConnectionPool(
                conninfo=_pg_conninfo(),
                min_size=1,
                max_size=3,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                open=False,
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            active_cp = saver

        graph = build_care_concierge_graph(
            session_factory=session_factory,
            classifier_fn=classifier_fn,
            toolbox=toolbox,
            checkpointer=active_cp,
        )
        config = {"configurable": {"thread_id": thread_id}}
        state_input: dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "thread_id": thread_id,
            "patient_id": patient_id,
            "request_id": request_id or f"req_{thread_id}",
        }
        return await graph.ainvoke(state_input, config=config)
    finally:
        if pool is not None:
            try:
                await pool.close()
            except Exception:
                pass