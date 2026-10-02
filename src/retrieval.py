"""Version-scoped pure vector retrieval and deterministic citation selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

from sqlalchemy import Float, Select, asc, literal, select
from sqlalchemy.orm import Session

from .db import Chunk, ContentType, DocumentVersion, DocumentVersionStatus
from .errors import ApplicationError, ErrorCode

DEFAULT_SECTION_DIVERSITY_LIMIT = 3
"""Max chunks kept from any single section before truncating to top_k.

Prevents one large section (e.g. several sibling sub-schemas) from
consuming every result slot and crowding out other, potentially more
relevant, sections.
"""

_CANDIDATE_POOL_MULTIPLIER = 3
"""How much larger a pool to fetch from the database before diversity
capping, so capping has enough candidates to promote in place of the ones
it drops."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A ranked chunk returned by the vector query."""

    chunk_id: UUID
    stable_chunk_id: str
    section_path: str
    content_type: ContentType
    content: str
    score: float
    rank: int
    selected_for_context: bool


@dataclass(frozen=True, slots=True)
class Citation:
    """A deterministic source group shown to the user."""

    section_path: str
    chunks: tuple[RetrievedChunk, ...]
    display_order: int


def build_vector_search_statement(query_vector: list[float], top_k: int) -> Select[Any]:
    """Build the v1 active-version vector query with stable tie-breaking."""

    if not query_vector:
        raise ValueError("query_vector must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    distance = Chunk.embedding.op("<=>")(query_vector)
    score = (literal(1.0, type_=Float) - distance).label("score")
    return (
        select(
            Chunk.id,
            Chunk.chunk_id,
            Chunk.section_path,
            Chunk.content_type,
            Chunk.content,
            score,
        )
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(DocumentVersion.status == DocumentVersionStatus.ACTIVE)
        .order_by(asc(distance), asc(Chunk.chunk_id))
        .limit(top_k)
    )


def cap_results_per_section(
    results: list[RetrievedChunk], max_per_section: int
) -> list[RetrievedChunk]:
    """Keep at most ``max_per_section`` highest-ranked chunks per section_path."""

    if max_per_section <= 0:
        raise ValueError("max_per_section must be positive")
    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for result in sorted(results, key=lambda item: item.rank):
        count = counts.get(result.section_path, 0)
        if count >= max_per_section:
            continue
        counts[result.section_path] = count + 1
        kept.append(result)
    return kept


def search_active_chunks(
    session: Session,
    query_vector: list[float],
    top_k: int,
    similarity_threshold: float | None = None,
    max_per_section: int | None = DEFAULT_SECTION_DIVERSITY_LIMIT,
) -> list[RetrievedChunk]:
    """Run v1 vector retrieval and mark candidates selected for model context.

    When ``max_per_section`` is set, a larger pool is fetched first and capped
    per section_path before truncating to ``top_k``, so a single dominant
    section cannot occupy every result slot.
    """

    if similarity_threshold is not None and not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    pool_size = top_k * _CANDIDATE_POOL_MULTIPLIER if max_per_section else top_k
    rows = session.execute(build_vector_search_statement(query_vector, pool_size)).all()
    candidates: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        candidates.append(
            RetrievedChunk(
                chunk_id=row.id,
                stable_chunk_id=row.chunk_id,
                section_path=row.section_path,
                content_type=row.content_type,
                content=row.content,
                score=float(row.score),
                rank=rank,
                selected_for_context=False,
            )
        )
    if max_per_section:
        candidates = cap_results_per_section(candidates, max_per_section)
    candidates = candidates[:top_k]

    return [
        replace(
            candidate,
            rank=new_rank,
            selected_for_context=(
                similarity_threshold is None or candidate.score >= similarity_threshold
            ),
        )
        for new_rank, candidate in enumerate(candidates, start=1)
    ]


def group_citations(results: list[RetrievedChunk]) -> list[Citation]:
    """Group selected results by section path while preserving rank order."""

    groups: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for result in sorted(results, key=lambda item: item.rank):
        if not result.selected_for_context:
            continue
        if result.section_path not in groups:
            groups[result.section_path] = []
            order.append(result.section_path)
        groups[result.section_path].append(result)
    return [
        Citation(section_path=section_path, chunks=tuple(groups[section_path]), display_order=index)
        for index, section_path in enumerate(order, start=1)
    ]


def document_not_ready_error() -> ApplicationError:
    """Return the stable error used when no active document version exists."""

    return ApplicationError(
        ErrorCode.DOCUMENT_NOT_READY,
        "no active document version is available",
    )
