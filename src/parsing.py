"""Markdown parsing pipeline built from the specified LlamaIndex components."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.llms import LLM
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
)


def load_markdown_document(source_path: Path) -> Document:
    """Read one Markdown source document without using an implicit model."""

    documents = SimpleDirectoryReader(input_files=[str(source_path)]).load_data()
    if len(documents) != 1:
        raise ValueError("the configured source path must resolve to exactly one document")
    return documents[0]


def parse_markdown_document(
    source_path: Path,
    llm: LLM,
    chunk_size_tokens: int = 1024,
    chunk_overlap_tokens: int = 80,
) -> list[Document]:
    """Parse Markdown chapters and elements with an explicitly supplied LLM.

    The element parser is intentionally not allowed to discover a global default
    LLM because table summaries are part of the ingestion contract.
    """

    if llm is None:
        raise ValueError("an explicit LLM is required for Markdown element parsing")
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be between 0 and chunk_size_tokens")

    document = load_markdown_document(source_path)
    chapter_parser = MarkdownNodeParser()
    chapter_nodes = chapter_parser.get_nodes_from_documents([document])
    element_parser = MarkdownElementNodeParser(
        llm=llm,
        nested_node_parser=SentenceSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
        ),
        show_progress=False,
    )
    return element_parser.get_nodes_from_documents(chapter_nodes)


def parse_markdown_chapters(source_path: Path) -> list[Document]:
    """Parse only chapter boundaries for validation that does not call an LLM."""

    document = load_markdown_document(source_path)
    return MarkdownNodeParser().get_nodes_from_documents([document])
