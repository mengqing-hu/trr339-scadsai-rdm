"""Pure ingestion utilities used by the document indexing workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from llama_index.core.schema import BaseNode, IndexNode, MetadataMode
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from .db import (
    Chunk,
    ContentType,
    Document,
    DocumentVersion,
    DocumentVersionStatus,
    IngestionJob,
    IngestionJobStatus,
    utc_now,
)
from .errors import ApplicationError, ErrorCode
from .parsing import parse_markdown_document
from .scads_client import ScadsClient
from .settings import AppSettings


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """Parser output before persistence and embedding."""

    section_path: str
    content_type: ContentType
    content: str
    retrieval_text: str
    token_count: int
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class ValidatedChunk:
    """Chunk input enriched with deterministic content identifiers."""

    chunk_id: str
    content_hash: str
    section_path: str
    content_type: ContentType
    content: str
    retrieval_text: str
    token_count: int
    embedding: list[float]
    embedding_dimension: int


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_chunk_id(
    document_id: str,
    section_path: str,
    content_type: ContentType,
    content_hash: str,
    occurrence_index: int,
) -> str:
    """Build a deterministic chunk identifier from normalized semantic inputs."""

    if occurrence_index < 0:
        raise ValueError("occurrence_index must not be negative")
    identity = "|".join(
        (
            document_id.strip(),
            section_path.strip(),
            content_type.value,
            content_hash,
            str(occurrence_index),
        )
    )
    return f"chunk_{sha256_text(identity)[:32]}"


def validate_source_path(source_path: Path, allowed_root: Path) -> Path:
    """Resolve and validate a Markdown source path within the configured root."""

    resolved_root = allowed_root.expanduser().resolve()
    resolved_source = source_path.expanduser().resolve()
    try:
        resolved_source.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("source path is outside the configured data root") from exc
    if resolved_source.suffix.lower() != ".md":
        raise ValueError("source path must use the .md extension")
    if not resolved_source.is_file():
        raise FileNotFoundError(f"source file does not exist: {resolved_source}")
    return resolved_source


def validate_chunk_inputs(
    document_id: str,
    chunks: list[ChunkInput],
    expected_embedding_dimension: int | None = None,
) -> list[ValidatedChunk]:
    """Validate parser output and assign stable IDs before database persistence."""

    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if not chunks:
        raise ValueError("index must contain at least one chunk")

    occurrences: dict[tuple[str, ContentType], int] = {}
    validated: list[ValidatedChunk] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        section_path = chunk.section_path.strip()
        if not section_path:
            raise ValueError("section_path must not be empty")
        if not chunk.content.strip():
            raise ValueError("chunk content must not be empty")
        if not chunk.retrieval_text.strip():
            raise ValueError("retrieval_text must not be empty")
        if chunk.token_count < 0:
            raise ValueError("token_count must not be negative")
        if not chunk.embedding:
            raise ValueError("embedding must not be empty")
        if any(not isinstance(value, (int, float)) for value in chunk.embedding):
            raise ValueError("embedding values must be numeric")

        dimension = len(chunk.embedding)
        if expected_embedding_dimension is not None and dimension != expected_embedding_dimension:
            raise ValueError("embedding dimension does not match configured dimension")

        content_hash = sha256_text(chunk.content)
        occurrence_key = (section_path, chunk.content_type)
        occurrence_index = occurrences.get(occurrence_key, 0)
        occurrences[occurrence_key] = occurrence_index + 1
        chunk_id = stable_chunk_id(
            document_id,
            section_path,
            chunk.content_type,
            content_hash,
            occurrence_index,
        )
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate stable chunk id: {chunk_id}")
        seen_ids.add(chunk_id)
        validated.append(
            ValidatedChunk(
                chunk_id=chunk_id,
                content_hash=content_hash,
                section_path=section_path,
                content_type=chunk.content_type,
                content=chunk.content,
                retrieval_text=chunk.retrieval_text,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                embedding_dimension=dimension,
            )
        )
    return validated


def source_error(error: Exception) -> ApplicationError:
    """Convert a source validation exception into a safe application error."""

    if isinstance(error, FileNotFoundError):
        return ApplicationError(ErrorCode.NOT_FOUND, "source document was not found", "source_path")
    return ApplicationError(ErrorCode.INVALID_INPUT, str(error), "source_path")


_MAX_HEADING_LINE_LENGTH = 120
_SENTENCE_ENDINGS = (".", "!", "?", ":", ",", ";")


def resolve_section_path(header_path: str | None, content: str) -> str:
    """Combine a node's ancestor header path with its own heading, when present.

    ``MarkdownNodeParser`` sets ``header_path`` metadata to the path of ANCESTOR
    headers only; a node's own heading (with the leading ``#`` markers stripped)
    is instead left as the first line of its content, separated from the body
    by a blank line. Using ``header_path`` alone therefore attributes a
    section's own text to its parent section. This reattaches that heading
    when the content shape indicates one is present, and otherwise falls back
    to the ancestor path unchanged (e.g. for split continuation chunks that do
    not start with a heading).
    """

    base = header_path or "/"
    heading, separator, rest = content.partition("\n\n")
    heading = heading.strip()
    if (
        not separator
        or not heading
        or not rest.strip()
        or "\n" in heading
        or len(heading) > _MAX_HEADING_LINE_LENGTH
        or heading.endswith(_SENTENCE_ENDINGS)
    ):
        return base
    return f"{base.rstrip('/')}/{heading}/"


def node_to_chunk_input(node: BaseNode) -> ChunkInput | None:
    """Convert a LlamaIndex node into the persisted text/table representation."""

    if isinstance(node, IndexNode):
        return None
    metadata = node.metadata
    table_summary = metadata.get("table_summary")
    if table_summary:
        raw_table = _extract_raw_table_markdown(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )
        return ChunkInput(
            section_path=str(metadata.get("header_path") or "/"),
            content_type=ContentType.TABLE,
            content=str(raw_table),
            retrieval_text=str(table_summary),
            token_count=max(1, len(str(raw_table).split())),
            embedding=[],
        )
    content = node.get_content(metadata_mode=MetadataMode.NONE)
    return ChunkInput(
        section_path=resolve_section_path(metadata.get("header_path"), content),
        content_type=ContentType.TEXT,
        content=content,
        retrieval_text=content,
        token_count=max(1, len(content.split())),
        embedding=[],
    )


def _extract_raw_table_markdown(content: str) -> str:
    """Extract the original Markdown table from a LlamaIndex table node.

    LlamaIndex stores the generated table summary and serialized table together
    in the node text. The metadata field named ``table_df`` is a dictionary
    serialization for regular tables, so it cannot be used as source content.
    """

    lines = content.splitlines()
    table_start = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("|")),
        None,
    )
    if table_start is None:
        raise ValueError("table node does not contain raw Markdown table content")

    table_lines: list[str] = []
    for line in lines[table_start:]:
        if not line.lstrip().startswith("|"):
            break
        table_lines.append(line)
    table = "\n".join(table_lines).strip()
    if not table:
        raise ValueError("table node contains an empty Markdown table")
    return table


def _build_document_id(source_path: Path) -> str:
    return source_path.stem.lower().replace("_", "-").replace(" ", "-")


def ingest_document(
    settings: AppSettings,
    session_factory: sessionmaker[Session],
    llm: Any,
    *,
    source_path: Path | None = None,
) -> UUID:
    """Build, validate, and atomically activate one document version."""

    source = validate_source_path(source_path or settings.source_path, settings.allowed_data_root)
    document_id = _build_document_id(source)
    document_hash = sha256_text(source.read_text(encoding="utf-8"))

    with session_factory() as session:
        document = session.scalar(select(Document).where(Document.document_id == document_id))
        if document is None:
            document = Document(document_id=document_id, source_path=str(source))
            session.add(document)
            session.flush()
        existing = session.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.document_id == document.id,
                IngestionJob.document_hash == document_hash,
                IngestionJob.status == IngestionJobStatus.SUCCEEDED,
            )
            .order_by(IngestionJob.created_at.desc())
        )
        if existing is not None:
            return existing.id
        job = IngestionJob(
            document_id=document.id,
            status=IngestionJobStatus.RUNNING,
            source_path=str(source),
            document_hash=document_hash,
            started_at=utc_now(),
        )
        version = DocumentVersion(
            document_id=document.id,
            document_hash=document_hash,
            status=DocumentVersionStatus.BUILDING,
        )
        session.add_all([job, version])
        session.flush()
        document_pk = document.id
        job.document_version_id = version.id
        session.commit()
        job_id = job.id
        version_id = version.id

    try:
        nodes = parse_markdown_document(
            source,
            llm,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        chunk_inputs = [node_to_chunk_input(node) for node in nodes]
        chunk_inputs = [chunk for chunk in chunk_inputs if chunk is not None]
        if not chunk_inputs:
            raise ValueError("parser produced no persistable chunks")

        for start in range(0, len(chunk_inputs), settings.embedding_batch_size):
            batch = chunk_inputs[start : start + settings.embedding_batch_size]
            embeddings = ScadsClient(settings).embed([chunk.retrieval_text for chunk in batch])
            hydrated = [
                ChunkInput(
                    section_path=chunk.section_path,
                    content_type=chunk.content_type,
                    content=chunk.content,
                    retrieval_text=chunk.retrieval_text,
                    token_count=chunk.token_count,
                    embedding=vector,
                )
                for chunk, vector in zip(batch, embeddings.vectors, strict=True)
            ]
            validated = validate_chunk_inputs(
                document_id,
                hydrated,
                expected_embedding_dimension=settings.embedding_dimension,
            )
            with session_factory() as session:
                session.add_all(
                    [
                        Chunk(
                            document_version_id=version_id,
                            chunk_id=item.chunk_id,
                            section_path=item.section_path,
                            content_type=item.content_type,
                            content=item.content,
                            retrieval_text=item.retrieval_text,
                            content_hash=item.content_hash,
                            token_count=item.token_count,
                            embedding=item.embedding,
                            embedding_model=embeddings.model,
                            embedding_dimension=item.embedding_dimension,
                        )
                        for item in validated
                    ]
                )
                session.commit()

        with session_factory() as session:
            chunk_count = session.scalar(
                select(Chunk.id).where(Chunk.document_version_id == version_id).limit(1)
            )
            if chunk_count is None:
                raise ValueError("index validation found no chunks")
            session.execute(
                update(DocumentVersion)
                .where(
                    DocumentVersion.document_id == document_pk,
                    DocumentVersion.status == DocumentVersionStatus.ACTIVE,
                )
                .values(status=DocumentVersionStatus.SUPERSEDED)
            )
            session.execute(
                update(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .values(status=DocumentVersionStatus.ACTIVE, activated_at=utc_now())
            )
            session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(status=IngestionJobStatus.SUCCEEDED, completed_at=utc_now())
            )
            session.commit()
        return job_id
    except Exception as error:
        with session_factory() as session:
            session.execute(
                update(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .values(status=DocumentVersionStatus.FAILED)
            )
            session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(
                    status=IngestionJobStatus.FAILED,
                    error_code=ErrorCode.INGESTION_VALIDATION_FAILED,
                    error_message=str(error)[:2000],
                    completed_at=utc_now(),
                )
            )
            session.commit()
        raise
