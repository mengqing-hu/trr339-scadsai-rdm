"""Transactional persistence helpers for conversations and answer runs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import (
    AnswerRun,
    AnswerRunStatus,
    Conversation,
    ConversationSummary,
    Message,
    MessageCitation,
    MessageRole,
    RetrievalResult,
    RouteType,
    User,
    utc_now,
)
from .errors import ApplicationError, ErrorCode


class ChatServiceError(RuntimeError):
    """Expected chat persistence failure with a stable application error."""

    def __init__(self, application_error: ApplicationError) -> None:
        super().__init__(application_error.message)
        self.application_error = application_error


def create_or_resume_conversation(
    session: Session,
    session_id: str,
    user_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> Conversation:
    """Create a conversation or validate that an existing one belongs to the caller."""

    if not session_id.strip():
        raise ChatServiceError(
            ApplicationError(ErrorCode.REQUIRED_FIELD, "session_id must not be empty", "session_id")
        )
    if conversation_id is not None:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "conversation was not found"))
        if conversation.session_id != session_id or conversation.user_id != user_id:
            raise ChatServiceError(ApplicationError(ErrorCode.ACCESS_DENIED, "conversation is not accessible"))
        return conversation

    conversation = Conversation(session_id=session_id, user_id=user_id)
    session.add(conversation)
    session.flush()
    return conversation


def append_user_message(session: Session, conversation_id: UUID, content: str) -> Message:
    """Append a user message using a session lock and a monotonic sequence."""

    if not content.strip():
        raise ChatServiceError(
            ApplicationError(ErrorCode.REQUIRED_FIELD, "question must not be empty", "question")
        )
    conversation = session.execute(
        select(Conversation).where(Conversation.id == conversation_id).with_for_update()
    ).scalar_one_or_none()
    if conversation is None:
        raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "conversation was not found"))
    max_sequence = session.scalar(
        select(func.max(Message.sequence_number)).where(Message.conversation_id == conversation_id)
    )
    message = Message(
        conversation_id=conversation_id,
        sequence_number=(max_sequence or 0) + 1,
        role=MessageRole.USER,
        content=content,
    )
    session.add(message)
    conversation.last_active_at = utc_now()
    session.flush()
    return message


def start_answer_run(
    session: Session,
    user_message: Message,
    retrieval_query: str,
    route_type: RouteType = RouteType.DOCUMENT_RAG,
    document_version_id: UUID | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    prompt_version: str | None = None,
) -> AnswerRun:
    """Create a running answer record for one user message."""

    if user_message.role is not MessageRole.USER:
        raise ChatServiceError(ApplicationError(ErrorCode.INVALID_INPUT, "message must be a user message"))
    if not retrieval_query.strip():
        raise ChatServiceError(
            ApplicationError(ErrorCode.REQUIRED_FIELD, "retrieval_query must not be empty", "retrieval_query")
        )
    existing = session.scalar(
        select(AnswerRun).where(AnswerRun.user_message_id == user_message.id)
    )
    if existing is not None:
        raise ChatServiceError(
            ApplicationError(ErrorCode.DUPLICATE_OPERATION, "answer run already exists for this message")
        )
    run = AnswerRun(
        user_message_id=user_message.id,
        document_version_id=document_version_id,
        retrieval_query=retrieval_query,
        route_type=route_type,
        status=AnswerRunStatus.RUNNING,
        chat_model=chat_model,
        embedding_model=embedding_model,
        prompt_version=prompt_version,
    )
    session.add(run)
    session.flush()
    return run


def complete_answer_run(
    session: Session,
    run_id: UUID,
    content: str,
    retrieval_result_ids: list[UUID],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    retrieval_latency_ms: int | None = None,
    generation_latency_ms: int | None = None,
    total_latency_ms: int | None = None,
) -> AnswerRun:
    """Persist an assistant message and final citations atomically with the run."""

    if not content.strip():
        raise ChatServiceError(
            ApplicationError(ErrorCode.GENERATION_FAILED, "assistant content must not be empty")
        )
    run = session.execute(select(AnswerRun).where(AnswerRun.id == run_id).with_for_update()).scalar_one_or_none()
    if run is None:
        raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "answer run was not found"))
    if run.status is not AnswerRunStatus.RUNNING:
        raise ChatServiceError(ApplicationError(ErrorCode.INVALID_STATE, "answer run is not running"))

    candidates = session.scalars(
        select(RetrievalResult).where(RetrievalResult.id.in_(retrieval_result_ids))
    ).all()
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    if any(candidate_id not in candidate_by_id for candidate_id in retrieval_result_ids):
        raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "retrieval result was not found"))
    if any(candidate.answer_run_id != run.id for candidate in candidates):
        raise ChatServiceError(ApplicationError(ErrorCode.INVALID_INPUT, "citation belongs to another run"))
    if any(not candidate.selected_for_context for candidate in candidates):
        raise ChatServiceError(ApplicationError(ErrorCode.INVALID_INPUT, "citation was not selected for context"))

    max_sequence = session.scalar(
        select(func.max(Message.sequence_number)).where(
            Message.conversation_id == session.scalar(
                select(Message.conversation_id).where(Message.id == run.user_message_id)
            )
        )
    )
    conversation_id = session.scalar(
        select(Message.conversation_id).where(Message.id == run.user_message_id)
    )
    if conversation_id is None:
        raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "user message was not found"))
    assistant = Message(
        conversation_id=conversation_id,
        sequence_number=(max_sequence or 0) + 1,
        role=MessageRole.ASSISTANT,
        content=content,
    )
    session.add(assistant)
    session.flush()
    for display_order, result_id in enumerate(retrieval_result_ids, start=1):
        session.add(
            MessageCitation(
                message_id=assistant.id,
                retrieval_result_id=result_id,
                display_order=display_order,
            )
        )
    run.assistant_message_id = assistant.id
    run.status = AnswerRunStatus.SUCCEEDED
    run.input_tokens = input_tokens
    run.output_tokens = output_tokens
    run.total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    run.retrieval_latency_ms = retrieval_latency_ms
    run.generation_latency_ms = generation_latency_ms
    run.total_latency_ms = total_latency_ms
    run.completed_at = utc_now()
    session.flush()
    return run


def fail_answer_run(
    session: Session,
    run_id: UUID,
    error_code: str,
    *,
    retrieval_latency_ms: int | None = None,
    generation_latency_ms: int | None = None,
    total_latency_ms: int | None = None,
) -> AnswerRun:
    """Mark a run failed without creating an assistant message or citation."""

    run = session.execute(select(AnswerRun).where(AnswerRun.id == run_id).with_for_update()).scalar_one_or_none()
    if run is None:
        raise ChatServiceError(ApplicationError(ErrorCode.NOT_FOUND, "answer run was not found"))
    if run.status is not AnswerRunStatus.RUNNING:
        raise ChatServiceError(ApplicationError(ErrorCode.INVALID_STATE, "answer run is not running"))
    run.status = AnswerRunStatus.FAILED
    run.error_code = error_code
    run.retrieval_latency_ms = retrieval_latency_ms
    run.generation_latency_ms = generation_latency_ms
    run.total_latency_ms = total_latency_ms
    run.completed_at = utc_now()
    session.flush()
    return run
