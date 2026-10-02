"""OpenAI-compatible ScaDS.AI client with response validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from .errors import ApplicationError, ErrorCode
from .settings import AppSettings


class EmbeddingsAPI(Protocol):
    def create(self, *, model: str, input: list[str]) -> Any: ...


class ChatCompletionsAPI(Protocol):
    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any: ...


class OpenAICompatibleClient(Protocol):
    embeddings: EmbeddingsAPI
    chat: Any


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model: str
    dimension: int
    usage: int | None


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class ScadsClientError(RuntimeError):
    """Error raised after an external ScaDS.AI response fails validation."""

    def __init__(self, application_error: ApplicationError) -> None:
        super().__init__(application_error.message)
        self.application_error = application_error


class ScadsClient:
    """Small explicit client boundary for embeddings and chat completions."""

    def __init__(self, settings: AppSettings, client: OpenAICompatibleClient | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.scads_api_key.get_secret_value(),
            base_url=settings.scads_api_url,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ScadsClientError(
                ApplicationError(ErrorCode.INVALID_INPUT, "embedding input must contain non-empty text")
            )
        try:
            response = self.client.embeddings.create(model=self.settings.embedding_model, input=texts)
        except Exception as exc:
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding service request failed")
            ) from exc

        vectors = self._read_vectors(response, len(texts))
        dimension = len(vectors[0])
        configured_dimension = self.settings.embedding_dimension
        if configured_dimension is not None and dimension != configured_dimension:
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding dimension mismatch")
            )
        if any(len(vector) != dimension for vector in vectors):
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding response has inconsistent dimensions")
            )
        usage = self._read_usage_total(response)
        return EmbeddingBatch(vectors=vectors, model=self.settings.embedding_model, dimension=dimension, usage=usage)

    def chat(self, messages: list[dict[str, str]]) -> ChatCompletion:
        if not messages or any(not message.get("content", "").strip() for message in messages):
            raise ScadsClientError(
                ApplicationError(ErrorCode.INVALID_INPUT, "chat messages must contain non-empty content")
            )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise ScadsClientError(
                ApplicationError(ErrorCode.GENERATION_FAILED, "chat service request failed")
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ScadsClientError(
                ApplicationError(ErrorCode.GENERATION_FAILED, "chat service returned empty content")
            )
        usage = getattr(response, "usage", None)
        return ChatCompletion(
            content=content,
            model=getattr(response, "model", self.settings.chat_model),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def _read_vectors(response: Any, expected_count: int) -> list[list[float]]:
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != expected_count:
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding response count mismatch")
            )
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors = [getattr(item, "embedding", None) for item in ordered]
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding response contains invalid vectors")
            )
        if any(any(not isinstance(value, (int, float)) for value in vector) for vector in vectors):
            raise ScadsClientError(
                ApplicationError(ErrorCode.EMBEDDING_FAILED, "embedding response contains non-numeric values")
            )
        return [[float(value) for value in vector] for vector in vectors]

    @staticmethod
    def _read_usage_total(response: Any) -> int | None:
        usage = getattr(response, "usage", None)
        value = getattr(usage, "total_tokens", None)
        return value if isinstance(value, int) and value >= 0 else None
