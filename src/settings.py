"""Typed application settings loaded from environment variables and YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Validated runtime settings for the application and worker processes."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    scads_api_url: str = Field(
        validation_alias=AliasChoices("SCADS_API_URL", "scads_api_url")
    )
    scads_api_key: SecretStr = Field(
        validation_alias=AliasChoices("SCADS_API_KEY", "scads_api_key")
    )
    database_url: str = Field(
        validation_alias=AliasChoices("DATABASE_URL", "database_url")
    )

    chat_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_configuration_version: str = "v1"
    parser_configuration_version: str = "v1"
    table_summary_prompt_version: str = "v1"
    table_summary_model: str | None = None
    embedding_dimension: int | None = None

    source_path: Path = Path("data/Manual-RDM.md")
    allowed_data_root: Path = Path("data")
    chunk_size_tokens: int = 1024
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 10
    retrieval_similarity_threshold: float | None = None
    memory_token_budget: int = 4096
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    max_question_chars: int = 8000
    max_feedback_comment_chars: int = 2000
    embedding_batch_size: int = 32

    @field_validator(
        "chunk_size_tokens",
        "memory_token_budget",
        "max_question_chars",
        "embedding_batch_size",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_non_negative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must not be negative")
        return value

    @field_validator("retrieval_top_k", "max_retries")
    @classmethod
    def validate_non_negative_or_positive(cls, value: int, info: Any) -> int:
        minimum = 1 if info.field_name == "retrieval_top_k" else 0
        if value < minimum:
            raise ValueError(f"value must be at least {minimum}")
        return value

    @field_validator("retrieval_similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("retrieval similarity threshold must be between 0 and 1")
        return value

    @field_validator("table_summary_model", mode="before")
    @classmethod
    def default_table_summary_model(cls, value: str | None) -> str:
        return value or "openai/gpt-oss-120b"


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load non-sensitive settings from a YAML mapping."""

    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return values


def load_settings(config_path: Path = Path("config/config.yaml")) -> AppSettings:
    """Load YAML defaults while allowing environment variables to override them."""

    yaml_values = load_yaml_config(config_path)
    environment_values: dict[str, Any] = {}
    for field_name in AppSettings.model_fields:
        environment_name = field_name.upper()
        if environment_name in os.environ:
            environment_values[field_name] = os.environ[environment_name]
    return AppSettings(**yaml_values, **environment_values)
