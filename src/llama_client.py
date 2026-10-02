"""LlamaIndex adapters for the ScaDS.AI OpenAI-compatible API."""

from __future__ import annotations

from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI

from .settings import AppSettings


class ScadsOpenAI(OpenAI):
    """Use LlamaIndex OpenAI transport without its fixed OpenAI model catalog."""

    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata for a custom ScaDS.AI chat model name."""

        return LLMMetadata(
            context_window=131072,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )

    def _should_use_structure_outputs(self) -> bool:
        """Use JSON Schema responses instead of unsupported tool_choice calls."""

        return True


def create_scads_llm(settings: AppSettings) -> ScadsOpenAI:
    """Create the explicit LLM used by Markdown table summarization."""

    return ScadsOpenAI(
        model=settings.table_summary_model or settings.chat_model,
        api_key=settings.scads_api_key.get_secret_value(),
        api_base=settings.scads_api_url,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
