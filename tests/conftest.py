import os
from pathlib import Path
import pytest
import pytest_asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.classifier import SafetyAndIntentClassifier
from app.config import get_settings
from app.database import create_engine_and_sessionmaker
from app.graph import build_care_concierge_graph
from app.tools import CareConciergeToolbox


@pytest_asyncio.fixture
async def db_session_factory():
    settings = get_settings()
    engine, session_factory = create_engine_and_sessionmaker(settings.DATABASE_URL)
    try:
        yield session_factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def pg_checkpointer():
    settings = get_settings()
    pg_conn_str = (
        settings.SYNC_DATABASE_URL
        .replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
    )
    pool = AsyncConnectionPool(
        conninfo=pg_conn_str,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver
    finally:
        await pool.close()


@pytest.fixture
def toolbox(db_session_factory):
    return CareConciergeToolbox(session_factory=db_session_factory)