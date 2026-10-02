#!/usr/bin/env python3
"""Ingest the configured RDM Markdown document into the active vector index."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.db import create_engine_from_url, create_session_factory
from src.ingestion import ingest_document
from src.llama_client import create_scads_llm
from src.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Markdown source path relative to the configured data root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    engine = create_engine_from_url(settings.database_url)
    session_factory = create_session_factory(engine)
    llm = create_scads_llm(settings)
    job_id = ingest_document(settings, session_factory, llm, source_path=args.source)
    print(f"ingestion_job_id={job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
