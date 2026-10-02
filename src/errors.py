"""Stable application errors and transport-independent operation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrorCode:
    """Stable machine-readable error codes from the application specification."""

    INVALID_INPUT = "INVALID_INPUT"
    REQUIRED_FIELD = "REQUIRED_FIELD"
    INVALID_ENUM = "INVALID_ENUM"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    NOT_FOUND = "NOT_FOUND"
    ACCESS_DENIED = "ACCESS_DENIED"
    INVALID_STATE = "INVALID_STATE"
    DUPLICATE_OPERATION = "DUPLICATE_OPERATION"
    CONCURRENT_REQUEST = "CONCURRENT_REQUEST"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DOCUMENT_NOT_READY = "DOCUMENT_NOT_READY"
    INGESTION_VALIDATION_FAILED = "INGESTION_VALIDATION_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    DATABASE_ERROR = "DATABASE_ERROR"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class ApplicationError:
    """A safe error payload that can be returned to a UI or API adapter."""

    code: str
    message: str
    field: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Transport-independent success or failure result."""

    request_id: str
    data: Any = None
    error: ApplicationError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "request_id": self.request_id, "data": self.data}
        return {
            "ok": False,
            "request_id": self.request_id,
            "error": self.error.as_dict() if self.error else None,
        }
