import logging
import math
import os
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocChunk, Society
from app.scoping import scoped_document_conditions

logger = logging.getLogger("concierge.rag")

# Dense-only cosine-similarity abstention threshold.  It is intentionally
# configurable so a deployment can re-tune it without a code change.
MIN_RETRIEVAL_SCORE = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.63"))
ABSTENTION_MESSAGE = (
    "I don't know based on the current clinic policies and your society annex. "
    "Would you like me to connect you with a nurse to assist further?"
)

# BGE models are trained with this instruction on the query side only.  The
# ingested passages stay unprefixed so existing committed vectors remain valid.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_INJECTION_PATTERNS = [
    r"(?im)^\s*(ignore|disregard|override)\s+(all\s+)?(previous|prior|system)\s+instructions.*$",
    r"(?im)^\s*(system|developer|assistant)\s*:\s*.*$",
    r"(?im)\b(call|invoke|execute)\s+(tool|book_service|escalate_emergency|log_symptom)\b.*$",
]

_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazy-load the dense semantic embedding transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from fastembed import TextEmbedding
        _EMBEDDING_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
    except Exception as exc:
        logger.error("embedding_model_unavailable error=%s", exc)
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _to_py_float(val: Any) -> float:
    """Convert numeric values (including numpy scalars) to pure Python float for msgpack."""
    if hasattr(val, "item"):
        val = val.item()
    return float(val)


def sanitize_retrieved_text(text: str) -> str:
    """Treat retrieved corpus text strictly as data, stripping instruction-like lines."""
    cleaned = text or ""
    for pat in _INJECTION_PATTERNS:
        cleaned = re.sub(pat, "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def embed_text(text: str, dim: int = 1536, *, query: bool = False) -> list[float]:
    """Generate a dense embedding with ``BAAI/bge-small-en-v1.5``.

    ``query=True`` adds the BGE retrieval instruction.  Passages are embedded
    without it.  A zero vector is returned only when the model is unavailable;
    callers must treat that as an abstention rather than as a search result.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return [0.0] * dim
    if query:
        clean_text = f"{_QUERY_INSTRUCTION}{clean_text}"

    model = get_embedding_model()
    if model is not None:
        try:
            raw_emb = list(model.embed([clean_text]))[0]
            emb_list = [float(x) for x in raw_emb]
            if len(emb_list) < dim:
                emb_list.extend([0.0] * (dim - len(emb_list)))
            elif len(emb_list) > dim:
                emb_list = emb_list[:dim]
            norm = math.sqrt(sum(x * x for x in emb_list))
            if norm > 0:
                emb_list = [x / norm for x in emb_list]
            return emb_list
        except Exception as exc:
            logger.error("embedding_failure error=%s", exc)

    return [0.0] * dim


def _abstention_response() -> dict[str, Any]:
    return {
        "abstained": True,
        "abstention_message": ABSTENTION_MESSAGE,
        "offer_nurse": True,
        "chunks": [],
        "citations": [],
    }


async def retrieve_corpus_chunks(
    session: AsyncSession,
    query: str,
    society: Society | None = None,
    *,
    society_id: str,
    top_k: int = 4,
    min_score: float = MIN_RETRIEVAL_SCORE,
    include_internal: bool = False,
) -> dict[str, Any]:
    """Dense pgvector retrieval with SQL society scoping, version filtering, and citations.

    The core ranker is dense-only: pgvector orders chunks by cosine distance to
    the query embedding.  There is no keyword, domain-term, annex, or coverage
    boost in this path.  Society applicability, superseded-version exclusion,
    and internal-note exclusion are all applied in the SQL ``WHERE`` clause
    before ranking, never as a Python post-filter.
    """
    if not society_id:
        raise ValueError("society_id is required for corpus retrieval.")
    if society is not None and society.id != society_id:
        raise ValueError(
            f"society_id={society_id!r} does not match supplied society {society.id!r}."
        )

    q_emb = embed_text(query, query=True)
    if not any(q_emb):
        logger.error("dense_retrieval_unavailable reason=embedding_model_unavailable")
        return _abstention_response()

    conditions = scoped_document_conditions(
        society_id=society_id, include_internal=include_internal
    )

    distance_expr = DocChunk.embedding.cosine_distance(q_emb)
    stmt = (
        select(DocChunk, distance_expr.label("distance"))
        .where(*conditions)
        .order_by(distance_expr.asc().nullslast())
        .limit(top_k)
    )

    # Scope and version are deliberately enforced by PostgreSQL above.  Do not
    # post-filter these rows in Python: doing so makes it too easy for a future
    # caller to accidentally retrieve another society's or a superseded chunk.
    rows = (await session.execute(stmt)).all()

    top_items: list[tuple[float, DocChunk]] = []
    for chunk, distance in rows:
        if distance is None:
            continue
        score = 1.0 - _to_py_float(distance)
        if not math.isfinite(score):
            continue
        score = max(0.0, min(1.0, score))
        if score >= min_score:
            top_items.append((score, chunk))

    if not top_items:
        return _abstention_response()

    chunks_out: list[dict[str, Any]] = []
    citations_out: list[dict[str, Any]] = []
    for score, ch in top_items:
        heading_path = ch.heading_path or (
            f"{ch.title} > {ch.section}" if ch.section and ch.section != ch.title else ch.title
        )
        safe_content = sanitize_retrieved_text(ch.content)
        chunks_out.append(
            {
                "chunk_id": str(ch.chunk_id),
                "doc_id": str(ch.doc_id),
                "title": str(ch.title),
                "section": str(ch.section),
                "heading_path": str(heading_path),
                "applies_to": str(ch.applies_to),
                "version": int(ch.version or 1),
                "score": _to_py_float(score),
                "content": str(safe_content),
            }
        )
        citations_out.append(
            {
                "doc_id": str(ch.doc_id),
                "title": str(ch.title),
                "section": str(ch.section),
                "heading_path": str(heading_path),
                "chunk_id": str(ch.chunk_id),
            }
        )

    return {
        "abstained": False,
        "abstention_message": None,
        "offer_nurse": False,
        "chunks": chunks_out,
        "citations": citations_out,
    }
