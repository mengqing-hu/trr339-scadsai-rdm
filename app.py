"""Streamlit entry point for the TRR 339 RDM Assistant."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import streamlit as st
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from src.chat import (
    ChatServiceError,
    append_user_message,
    complete_answer_run,
    create_or_resume_conversation,
    fail_answer_run,
    start_answer_run,
)
from src.db import (
    Chunk,
    DocumentVersion,
    DocumentVersionStatus,
    Message,
    MessageCitation,
    MessageRole,
    RetrievalResult,
    RouteType,
    create_engine_from_url,
    create_session_factory,
)
from src.errors import ErrorCode
from src.ingestion import ingest_document
from src.llama_client import create_scads_llm
from src.retrieval import group_citations, search_active_chunks
from src.scads_client import ScadsClient, ScadsClientError
from src.settings import AppSettings, load_settings


@st.cache_resource
def get_runtime() -> tuple[AppSettings, object, object, ScadsClient]:
    """Create process-wide settings, database session factory, and API client."""

    settings = load_settings()
    engine = create_engine_from_url(settings.database_url)
    session_factory = create_session_factory(engine)
    return settings, engine, session_factory, ScadsClient(settings)


def get_conversation_id(session_factory: object, session_id: str) -> UUID:
    if "conversation_id" in st.session_state:
        return st.session_state.conversation_id
    with session_factory() as session:
        conversation = create_or_resume_conversation(session, session_id)
        session.commit()
        st.session_state.conversation_id = conversation.id
        return conversation.id


def answer_question(
    settings: AppSettings,
    session_factory: object,
    client: ScadsClient,
    conversation_id: UUID,
    question: str,
) -> dict[str, object]:
    """Run one question through retrieval, generation, and persistence."""

    with session_factory() as session:
        user_message = append_user_message(session, conversation_id, question)
        active_version = session.scalar(
            select(DocumentVersion.id).where(DocumentVersion.status == DocumentVersionStatus.ACTIVE)
        )
        run = start_answer_run(
            session,
            user_message,
            question,
            route_type=RouteType.DOCUMENT_RAG,
            document_version_id=active_version,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            prompt_version="v1",
        )
        # Commit the request and running audit record before external calls so
        # failures can always be recorded without losing the user message.
        session.commit()
        request_started = time.perf_counter()
        retrieval_started = request_started
        generation_started: float | None = None
        retrieval_latency_ms: int | None = None
        generation_latency_ms: int | None = None

        def elapsed_ms(started: float) -> int:
            return max(0, int((time.perf_counter() - started) * 1000))

        try:
            if active_version is None:
                fail_answer_run(
                    session,
                    run.id,
                    ErrorCode.DOCUMENT_NOT_READY,
                    total_latency_ms=elapsed_ms(request_started),
                )
                session.commit()
                return {
                    "status": "failed",
                    "message": "No document index is available. Ingest Manual-RDM.md first.",
                }

            retrieval_started = time.perf_counter()
            embedding = client.embed([question])
            run.embedding_model = embedding.model
            results = search_active_chunks(
                session,
                embedding.vectors[0],
                settings.retrieval_top_k,
                settings.retrieval_similarity_threshold,
            )
            retrieval_latency_ms = elapsed_ms(retrieval_started)
            persisted_results: list[RetrievalResult] = []
            for result in results:
                persisted_results.append(
                    RetrievalResult(
                        answer_run_id=run.id,
                        chunk_id=result.chunk_id,
                        score=result.score,
                        rank=result.rank,
                        selected_for_context=result.selected_for_context,
                    )
                )
            session.add_all(persisted_results)
            session.flush()

            selected_results = [result for result in results if result.selected_for_context]
            if not selected_results:
                content = (
                    "No reliable information relevant to this question was found in the manual."
                )
                complete_answer_run(
                    session,
                    run.id,
                    content,
                    [],
                    retrieval_latency_ms=retrieval_latency_ms,
                    total_latency_ms=elapsed_ms(request_started),
                )
                session.commit()
                return {"status": "succeeded", "content": content, "citations": []}

            evidence = "\n\n".join(
                f"[{result.section_path}]\n{result.content}" for result in selected_results
            )
            generation_started = time.perf_counter()
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied RDM evidence. "
                            "If the evidence is insufficient, say so explicitly. "
                            "Do not invent citations or facts.\n\nEvidence:\n" + evidence
                        ),
                    },
                    {"role": "user", "content": question},
                ]
            )
            generation_latency_ms = elapsed_ms(generation_started)
            run.chat_model = response.model
            selected_ids = [persisted_results[result.rank - 1].id for result in selected_results]
            complete_answer_run(
                session,
                run.id,
                response.content,
                selected_ids,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=elapsed_ms(request_started),
            )
            citations = [
                {
                    "section_path": citation.section_path,
                    "content": "\n\n".join(chunk.content for chunk in citation.chunks),
                }
                for citation in group_citations(results)
            ]
            session.commit()
            return {"status": "succeeded", "content": response.content, "citations": citations}
        except ScadsClientError as error:
            if retrieval_latency_ms is None and retrieval_started != request_started:
                retrieval_latency_ms = elapsed_ms(retrieval_started)
            if generation_latency_ms is None and generation_started is not None:
                generation_latency_ms = elapsed_ms(generation_started)
            fail_answer_run(
                session,
                run.id,
                error.application_error.code,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=elapsed_ms(request_started),
            )
            session.commit()
            return {"status": "failed", "message": error.application_error.message}
        except ChatServiceError as error:
            if generation_latency_ms is None and generation_started is not None:
                generation_latency_ms = elapsed_ms(generation_started)
            session.rollback()
            try:
                fail_answer_run(
                    session,
                    run.id,
                    error.application_error.code,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    total_latency_ms=elapsed_ms(request_started),
                )
                session.commit()
            except Exception:
                session.rollback()
            return {"status": "failed", "message": error.application_error.message}
        except Exception:
            if generation_latency_ms is None and generation_started is not None:
                generation_latency_ms = elapsed_ms(generation_started)
            session.rollback()
            try:
                fail_answer_run(
                    session,
                    run.id,
                    ErrorCode.INTERNAL_ERROR,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    total_latency_ms=elapsed_ms(request_started),
                )
                session.commit()
            except Exception:
                session.rollback()
            return {
                "status": "failed",
                "message": "An internal error occurred while processing the question.",
            }


def render_history(session_factory: object, conversation_id: UUID) -> None:
    with session_factory() as session:
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_number.asc())
        ).all()
        for message in messages:
            with st.chat_message(message.role.value):
                st.markdown(message.content)
                if message.role is MessageRole.ASSISTANT:
                    citation_rows = session.execute(
                        select(MessageCitation, RetrievalResult, Chunk)
                        .join(
                            RetrievalResult,
                            RetrievalResult.id == MessageCitation.retrieval_result_id,
                        )
                        .join(Chunk, Chunk.id == RetrievalResult.chunk_id)
                        .where(MessageCitation.message_id == message.id)
                        .order_by(MessageCitation.display_order.asc())
                    ).all()
                    if citation_rows:
                        with st.expander("Sources"):
                            for citation, retrieval, chunk in citation_rows:
                                st.markdown(f"**{retrieval.rank}. {chunk.section_path}**")
                                st.caption(f"score={retrieval.score:.6f}")


def main() -> None:
    st.set_page_config(page_title="TRR 339 RDM Assistant", layout="wide")
    st.title("TRR 339 RDM Assistant")
    try:
        settings, _engine, session_factory, client = get_runtime()
    except Exception as error:
        st.error(f"Configuration or database initialization failed: {error}")
        st.stop()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    try:
        conversation_id = get_conversation_id(session_factory, st.session_state.session_id)
    except OperationalError:
        st.error(
            "Database connection failed. Confirm that PostgreSQL with pgvector is running "
            "and that DATABASE_URL uses the correct port."
        )
        st.info(
            "The default project database URL is postgresql+psycopg://rdm:rdm@localhost:5433/rdm"
        )
        st.stop()

    with st.sidebar:
        st.caption(f"conversation_id: {conversation_id}")
        if st.button("Rebuild Document Index", use_container_width=True):
            try:
                llm = create_scads_llm(settings)
                with st.spinner("Ingesting the document and rebuilding the index..."):
                    job_id = ingest_document(settings, session_factory, llm)
                st.success(f"Ingestion completed: {job_id}")
            except Exception as error:
                st.error(f"Ingestion failed: {error}")

    render_history(session_factory, conversation_id)
    question = st.chat_input("Ask a question about the RDM manual")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Retrieving sources and generating an answer..."):
            result = answer_question(settings, session_factory, client, conversation_id, question)
        with st.chat_message("assistant"):
            if result["status"] == "succeeded":
                st.markdown(str(result["content"]))
                for citation in result.get("citations", []):
                    with st.expander(f"Source: {citation['section_path']}"):
                        st.markdown(str(citation["content"]))
            else:
                st.warning(str(result["message"]))
        st.rerun()


if __name__ == "__main__":
    main()
