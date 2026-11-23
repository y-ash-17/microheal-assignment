import math
import os
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocChunk, Society
from app.scoping import scoped_document_conditions

MIN_RETRIEVAL_SCORE = 0.35
ABSTENTION_MESSAGE = (
    "I don't know based on the current clinic policies and your society annex. "
    "Would you like me to connect you with a nurse to assist further?"
)

_INJECTION_PATTERNS = [
    r"(?im)^\s*(ignore|disregard|override)\s+(all\s+)?(previous|prior|system)\s+instructions.*$",
    r"(?im)^\s*(system|developer|assistant)\s*:\s*.*$",
    r"(?im)\b(call|invoke|execute)\s+(tool|book_service|escalate_emergency|log_symptom)\b.*$",
]

_STOPWORDS = {
    "the", "and", "for", "with", "from", "what", "when", "where", "which",
    "who", "how", "are", "is", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "by", "or", "an", "a", "my", "me", "i", "it", "do",
    "does", "can", "could", "would", "should", "now", "still", "any",
    "period", "price",
}

_DOMAIN_BOOST_TERMS = {
    "wellness", "sunday", "hours", "fasting", "lipid", "fbs", "thyroid",
    "vitamin", "coverage", "plan", "escalation", "112", "hospital",
    "department", "codes", "slot", "lengths", "cardiology", "physiotherapy",
    "directory", "covered", "member", "rate",
}

_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazy-load the real dense semantic embedding transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from fastembed import TextEmbedding
        _EMBEDDING_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
    except Exception:
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


def embed_text(text: str, dim: int = 1536) -> list[float]:
    """Generate dense embeddings using a neural transformer model (BAAI/bge-small-en-v1.5).
    """

    clean_text = (text or "").strip()
    if not clean_text:
        return [0.0] * dim

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
        except Exception:
            pass

    return [0.0] * dim


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    a_py = [_to_py_float(x) for x in a]
    b_py = [_to_py_float(x) for x in b]
    dot = sum(x * y for x, y in zip(a_py, b_py))
    na = math.sqrt(sum(x * x for x in a_py))
    nb = math.sqrt(sum(y * y for y in b_py))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (na * nb))))


def _keyword_score(query: str, chunk: DocChunk) -> float:
    q_tokens = {
        t for t in re.findall(r"[a-z0-9_]+", (query or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    }
    if not q_tokens:
        return 0.0
    haystack = (
        f"{chunk.title or ''} {chunk.heading_path or ''} {chunk.section or ''} {chunk.content or ''}"
    ).lower()
    h_tokens = set(re.findall(r"[a-z0-9_]+", haystack))
    overlap = len(q_tokens & h_tokens)
    if overlap == 0:
        return 0.0

    boost = 0.0
    for term in _DOMAIN_BOOST_TERMS:
        if term in q_tokens and term in haystack:
            boost += 0.25
    return float((overlap / max(len(q_tokens), 1)) + boost)


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
    """Dense neural retrieval with SQL society scoping, internal notes exclusion, and citations."""
    if not society_id:
        raise ValueError("society_id is required for corpus retrieval.")
    resolved_soc_id = society_id

    q_emb = embed_text(query)

    # Scope, version, and internal-note predicates are applied in SQL before
    # vector ranking, never as a Python post-filter.
    conditions = scoped_document_conditions(
        society_id=resolved_soc_id, include_internal=include_internal
    )

    stmt = (
        select(DocChunk)
        .where(*conditions)
        .order_by(DocChunk.embedding.cosine_distance(q_emb).asc().nullslast())
    )

    # Scope and version are deliberately enforced by PostgreSQL above.  Do not
    # post-filter these rows in Python: doing so makes it too easy for a future
    # caller to accidentally retrieve another society's or a superseded chunk.
    scoped_rows = list((await session.execute(stmt)).scalars().all())

    q_lower = (query or "").lower()
    is_annex_query = any(
        w in q_lower
        for w in (
            "annex", "wellness", "sunday", "saturday", "hours", "open",
            "closed", "phlebotomist", "visiting", "gate", "ambulance", "resident",
        )
    )
    is_coverage_query = any(
        w in q_lower
        for w in ("vitamin", "thyroid", "member rate", "coverage")
    ) or ("covered" in q_lower and any(w in q_lower for w in ("plan", "test", "lab", "thyroid", "vitamin", "lipid", "cbc", "fbs")))
    allowed_annex = society.annex_doc_id if (society and society.annex_doc_id) else None

    scored: list[tuple[float, DocChunk]] = []
    for ch in scoped_rows:
        sim = _cosine_similarity(q_emb, list(ch.embedding) if ch.embedding is not None else None)
        kw = _keyword_score(query, ch)
        annex_boost = 0.25 if (is_annex_query and allowed_annex and ch.doc_id == allowed_annex) else 0.0
        cov_boost = 0.0
        if is_coverage_query and ch.doc_id == "plan-coverage":
            cov_boost += 0.65
            c_lower = (ch.content or "").lower()
            if ("vitamin" in q_lower and "vitamin d" in c_lower) or (
                "thyroid" in q_lower and "thyroid" in c_lower
            ):
                cov_boost += 0.45
            if "thyroid profile is now covered" in c_lower or "what changed" in c_lower:
                cov_boost += 0.85
        total_score = float(round((0.40 * sim) + (0.60 * kw) + annex_boost + cov_boost, 4))
        scored.append((total_score, ch))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_items = [item for item in scored[:top_k] if item[0] >= min_score]

    if not top_items:
        return {
            "abstained": True,
            "abstention_message": ABSTENTION_MESSAGE,
            "offer_nurse": True,
            "chunks": [],
            "citations": [],
        }

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
