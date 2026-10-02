"""SQLAlchemy models, database engine creation, and session helpers."""

from __future__ import annotations

import enum
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
    create_engine,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None


class PortableVector(TypeDecorator[list[float]]):
    """Use JSON for local SQLite tests and native VECTOR on PostgreSQL."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimension: int | None = None, **kwargs: Any) -> None:
        self.dimension = dimension
        super().__init__(**kwargs)


@compiles(PortableVector, "postgresql")
def compile_portable_vector(type_: PortableVector, compiler: Any, **kwargs: Any) -> str:
    dimension = f"({type_.dimension})" if type_.dimension else ""
    return f"VECTOR{dimension}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all application tables."""


class DocumentVersionStatus(str, enum.Enum):
    BUILDING = "building"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class IngestionJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnswerRunStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"


class RouteType(str, enum.Enum):
    CONVERSATION_LOOKUP = "conversation_lookup"
    CONVERSATION_SEMANTIC_SEARCH = "conversation_semantic_search"
    DOCUMENT_RAG = "document_rag"
    MIXED = "mixed"


class HistorySourceType(str, enum.Enum):
    MESSAGE = "message"
    SUMMARY = "summary"


class FeedbackRating(str, enum.Enum):
    HELPFUL = "helpful"
    UNHELPFUL = "unhelpful"


def enum_type(enum_class: type[enum.Enum]) -> Enum:
    return Enum(
        enum_class,
        values_callable=lambda values: [member.value for member in values],
        native_enum=False,
        validate_strings=True,
    )


def vector_type(dimension: int | None = None) -> Any:
    if Vector is not None:
        return Vector(dimension)
    return PortableVector(dimension)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        Index(
            "uq_document_versions_active",
            "document_id",
            unique=True,
            postgresql_where="status = 'active'",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False
    )
    document_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[DocumentVersionStatus] = mapped_column(
        enum_type(DocumentVersionStatus), nullable=False
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    external_subject: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False
    )
    document_version_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_versions.id", ondelete="RESTRICT")
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        enum_type(IngestionJobStatus), nullable=False
    )
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_hash: Mapped[str | None] = mapped_column(String(64))
    requested_by_user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_id", name="uq_chunks_version_chunk_id"),
        CheckConstraint("token_count >= 0", name="ck_chunks_token_count_non_negative"),
        CheckConstraint("embedding_dimension > 0", name="ck_chunks_embedding_dimension_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(String(255), nullable=False)
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[ContentType] = mapped_column(enum_type(ContentType), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(vector_type(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence_number", name="uq_messages_conversation_sequence"
        ),
        CheckConstraint("sequence_number >= 1", name="ck_messages_sequence_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("conversations.id", ondelete="RESTRICT"), nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[MessageRole] = mapped_column(enum_type(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AnswerRun(Base):
    __tablename__ = "answer_runs"
    __table_args__ = (
        UniqueConstraint("user_message_id", name="uq_answer_runs_user_message"),
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="ck_answer_runs_input"),
        CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="ck_answer_runs_output"),
        CheckConstraint("total_tokens IS NULL OR total_tokens >= 0", name="ck_answer_runs_total"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("messages.id", ondelete="RESTRICT"), nullable=False
    )
    assistant_message_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("messages.id", ondelete="RESTRICT")
    )
    document_version_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_versions.id", ondelete="RESTRICT")
    )
    retrieval_query: Mapped[str] = mapped_column(Text, nullable=False)
    route_type: Mapped[RouteType] = mapped_column(enum_type(RouteType), nullable=False)
    status: Mapped[AnswerRunStatus] = mapped_column(enum_type(AnswerRunStatus), nullable=False)
    chat_model: Mapped[str | None] = mapped_column(String(255))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    retrieval_latency_ms: Mapped[int | None] = mapped_column(Integer)
    generation_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (
        UniqueConstraint("answer_run_id", "rank", name="uq_retrieval_results_run_rank"),
        CheckConstraint("rank >= 1", name="ck_retrieval_results_rank_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    answer_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("answer_runs.id", ondelete="RESTRICT"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_for_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MessageCitation(Base):
    __tablename__ = "message_citations"
    __table_args__ = (
        UniqueConstraint("message_id", "display_order", name="uq_message_citations_order"),
        CheckConstraint("display_order >= 1", name="ck_message_citations_order_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("messages.id", ondelete="RESTRICT"), nullable=False
    )
    retrieval_result_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("retrieval_results.id", ondelete="RESTRICT"), nullable=False
    )
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("comment IS NULL OR length(trim(comment)) > 0", name="ck_feedback_comment"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("messages.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    rating: Mapped[FeedbackRating] = mapped_column(enum_type(FeedbackRating), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        CheckConstraint(
            "start_sequence_number >= 1", name="ck_conversation_summaries_start_positive"
        ),
        CheckConstraint(
            "end_sequence_number >= start_sequence_number",
            name="ck_conversation_summaries_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("conversations.id", ondelete="RESTRICT"), nullable=False
    )
    start_sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    end_sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class HistoryEmbedding(Base):
    __tablename__ = "history_embeddings"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    source_type: Mapped[HistorySourceType] = mapped_column(
        enum_type(HistorySourceType), nullable=False
    )
    source_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    embedding: Mapped[Any] = mapped_column(vector_type(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def create_engine_from_url(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with safe connection liveness defaults."""

    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **engine_kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the application session factory."""

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield a transaction-scoped session and roll back unhandled failures."""

    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
