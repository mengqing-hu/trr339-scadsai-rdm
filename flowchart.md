# TRR 339 RDM Assistant: File Guide and System Flow

## 1. Purpose

This document explains the responsibility of every project-owned file in the
current repository and shows how the application starts, indexes the RDM manual,
answers questions, and persists its results.

The application is a retrieval-augmented generation system with four main parts:

- A Streamlit user interface.
- A PostgreSQL database with the pgvector extension.
- A Markdown ingestion pipeline built with LlamaIndex.
- A ScaDS.AI OpenAI-compatible API for embeddings and answer generation.

## 2. System Overview

```mermaid
flowchart LR
    User[User] --> UI[Streamlit application<br/>app.py]

    UI --> Settings[Settings loader<br/>src/settings.py]
    UI --> Chat[Chat persistence<br/>src/chat.py]
    UI --> Retrieval[Vector retrieval<br/>src/retrieval.py]
    UI --> Client[ScaDS.AI client<br/>src/scads_client.py]
    UI --> Ingestion[Ingestion service<br/>src/ingestion.py]

    Script[Ingestion CLI<br/>scripts/ingest_documents.py] --> Ingestion
    Ingestion --> Parser[Markdown parsing<br/>src/parsing.py]
    Ingestion --> LlamaAdapter[LlamaIndex adapter<br/>src/llama_client.py]
    Ingestion --> Client

    Parser --> Manual[data/Manual-RDM.md]
    LlamaAdapter --> Scads[ScaDS.AI API]
    Client --> Scads

    Chat --> Database[(PostgreSQL and pgvector)]
    Retrieval --> Database
    Ingestion --> Database
    DatabaseConfig[compose.yaml] --> Database
    Schema[Alembic<br/>alembic.ini and migrations/env.py] --> Database
```

## 3. Repository File Guide

### 3.1 Root Files

| File | Responsibility |
| --- | --- |
| `.env` | Local runtime secrets and environment-specific values. It supplies the ScaDS.AI endpoint, API key, database URL, and optional port override. It must not be committed. |
| `.env.example` | Template showing the environment variables required by the application. Developers copy it to `.env` and provide real values. |
| `.gitignore` | Excludes secrets, local data, local configuration, documentation directories, caches, and generated Python artifacts from Git. |
| `README` | Primary setup and operating guide. It describes the project scope, database startup, schema migration, document ingestion, application startup, and verification commands. |
| `alembic.ini` | Alembic configuration. It points Alembic to `migrations/`, configures logging, and supplies a placeholder URL that `migrations/env.py` replaces at runtime. |
| `app.py` | Streamlit entry point and application orchestrator. It initializes settings and clients, creates conversations, handles questions, runs retrieval and generation, records results, renders citations, and exposes the document-index rebuild action. |
| `compose.yaml` | Defines the local `pgvector/pgvector:pg16` database container, credentials, persistent volume, health check, and host port mapping from `5433` to PostgreSQL port `5432`. |
| `environment.md` | Personal environment notes for creating a virtual environment and managing a Jupyter kernel. Some commands are historical and are not part of the application runtime. |
| `flowchart.md` | This file. It documents file responsibilities and end-to-end execution flows. |

### 3.2 Configuration and Source Data

| File | Responsibility |
| --- | --- |
| `config/config.yaml` | Non-secret application defaults, including model names, source path, chunk sizes, retrieval limits, timeouts, retry counts, and embedding batch size. Environment variables override matching values. |
| `data/Manual-RDM.md` | Canonical Markdown knowledge source indexed by the application. Its text, headings, and tables become searchable chunks. |

### 3.3 Command-Line Scripts

| File | Responsibility |
| --- | --- |
| `scripts/ingest_documents.py` | Thin command-line entry point for document ingestion. It parses `--source`, loads settings, creates database and LLM dependencies, invokes `ingest_document`, and prints the ingestion job ID. |

The script contains no indexing rules itself. The reusable implementation stays
in `src/ingestion.py`, allowing both the command line and Streamlit sidebar to
run the same ingestion process.

### 3.4 Application Modules

| File | Responsibility |
| --- | --- |
| `src/__init__.py` | Marks `src` as a Python package and identifies it as the TRR 339 RDM Assistant application package. |
| `src/settings.py` | Defines validated typed settings with Pydantic. It loads `.env`, reads YAML defaults, applies environment overrides, protects secret values, and validates numeric configuration. |
| `src/db.py` | Defines all SQLAlchemy enums, tables, constraints, relationships through foreign keys, pgvector column behavior, engine creation, session creation, and transaction helpers. |
| `src/errors.py` | Defines stable machine-readable error codes and transport-independent success or failure payloads. |
| `src/chat.py` | Implements transactional conversation persistence. It creates conversations, appends sequenced messages, starts answer runs, completes runs with citations and token metrics, and records failures. |
| `src/scads_client.py` | Wraps the ScaDS.AI OpenAI-compatible API. It requests embeddings and chat completions, validates response shapes, captures token usage, and converts external failures into stable application errors. |
| `src/llama_client.py` | Adapts the ScaDS.AI chat model to LlamaIndex. It supplies model metadata and enables structured output for Markdown table summarization. |
| `src/parsing.py` | Loads one Markdown file and parses it with LlamaIndex. It separates chapter structure, text chunks, and table elements using configured chunk size and overlap. |
| `src/ingestion.py` | Implements the versioned indexing workflow. It validates source paths, hashes documents, avoids duplicate ingestion, parses content, creates embeddings, validates chunks, stores them, and atomically activates a successful document version. |
| `src/retrieval.py` | Builds cosine-distance pgvector queries against the active document version, enforces section diversity, applies the optional similarity threshold, ranks results deterministically, and groups selected chunks into citations. |

### 3.5 Database Migration Files

| File | Responsibility |
| --- | --- |
| `migrations/env.py` | Connects Alembic to the application settings and SQLAlchemy metadata. It supports online and offline migration execution and enables type comparison. |

The current checkout has no file under `migrations/versions/` and no
`migrations/script.py.mako`. The database may continue to work if it was migrated
earlier, but a clean checkout cannot recreate or evolve the schema with Alembic
until the migration revision and template are restored or regenerated.



## 4. Configuration Flow

`src/settings.py` combines configuration from YAML and environment sources.
Secrets belong in `.env`; non-secret defaults belong in `config/config.yaml`.

```mermaid
flowchart TD
    Yaml[config/config.yaml<br/>non-secret defaults] --> Loader[load_settings]
    DotEnv[.env<br/>local secrets and overrides] --> Pydantic[Pydantic BaseSettings]
    ProcessEnv[Process environment variables] --> Loader
    Loader --> Pydantic
    Pydantic --> Validate{Values valid?}
    Validate -- No --> ConfigError[Stop with configuration error]
    Validate -- Yes --> Settings[AppSettings]
    Settings --> App[Streamlit application]
    Settings --> CLI[Ingestion CLI]
    Settings --> Alembic[Alembic migration environment]
```

Important precedence behavior:

1. `config/config.yaml` supplies base values.
2. Matching process environment variables are explicitly passed as overrides.
3. Pydantic also reads `.env` for values not already supplied.
4. Validation rejects invalid limits, overlaps, thresholds, and missing required
   connection values.

## 5. Local Startup Flow

```mermaid
flowchart TD
    Start[Developer starts local environment] --> Docker[docker compose up -d]
    Docker --> Postgres[(PostgreSQL with pgvector)]
    Postgres --> Health{Container healthy?}
    Health -- No --> FixDatabase[Inspect Docker and port 5433]
    Health -- Yes --> Migrate[alembic upgrade head]
    Migrate --> Schema{Migration revisions available?}
    Schema -- No --> RestoreMigration[Restore or generate migration revisions]
    Schema -- Yes --> Ingest[Run scripts/ingest_documents.py]
    Ingest --> Indexed{Active document version exists?}
    Indexed -- No --> FixIngestion[Inspect source, API, and ingestion job]
    Indexed -- Yes --> Streamlit[streamlit run app.py]
    Streamlit --> Browser[Open Streamlit URL]
```

The expected order is database, schema, document index, then application. The
Streamlit process can start before ingestion, but it will report that no active
document index is available when a user asks a question.

## 6. Document Ingestion Flow

Ingestion can start from either `scripts/ingest_documents.py` or the Streamlit
sidebar button. Both entry points call the same `ingest_document` function.

```mermaid
flowchart TD
    Trigger[CLI or Rebuild Document Index button] --> Load[Load settings and dependencies]
    Load --> ValidatePath[Validate Markdown source path]
    ValidatePath --> Hash[Calculate document SHA-256 hash]
    Hash --> Existing{Successful job with same hash?}

    Existing -- Yes --> ReturnExisting[Return existing ingestion job ID]
    Existing -- No --> CreateRecords[Create running ingestion job<br/>and building document version]

    CreateRecords --> Parse[Parse Markdown with LlamaIndex]
    Parse --> Split[Create text and table chunks]
    Split --> TableSummary[Generate table summaries through ScaDS.AI]
    TableSummary --> Batch[Process chunks in configured batches]
    Batch --> Embed[Request embeddings from ScaDS.AI]
    Embed --> ValidateChunks[Validate content, dimensions,<br/>hashes, and stable chunk IDs]
    ValidateChunks --> Store[Store chunks and vectors in pgvector]
    Store --> More{More batches?}
    More -- Yes --> Batch
    More -- No --> Verify{At least one chunk stored?}

    Verify -- Yes --> Supersede[Mark previous active version superseded]
    Supersede --> Activate[Mark new version active]
    Activate --> Succeed[Mark ingestion job succeeded]
    Succeed --> ReturnNew[Return new ingestion job ID]

    ValidatePath -. failure .-> Fail[Mark version and job failed]
    Parse -. failure .-> Fail
    TableSummary -. failure .-> Fail
    Embed -. failure .-> Fail
    ValidateChunks -. failure .-> Fail
    Store -. failure .-> Fail
    Verify -- No --> Fail
    Fail --> Preserve[Keep previous active version unchanged]
    Preserve --> Raise[Raise the ingestion error]
```

### Ingestion Guarantees

- Source files must be Markdown files inside the configured data root.
- A content hash makes successful ingestion idempotent.
- Chunk IDs are stable for the same document content and section position.
- Embedding dimensions must be consistent.
- Chunks are written in batches to limit request and transaction size.
- The new version becomes active only after all validation succeeds.
- A failed build does not replace the previous active version.

## 7. Question-Answer Flow

```mermaid
flowchart TD
    Open[User opens Streamlit] --> Runtime[Create cached settings, engine,<br/>session factory, and ScaDS.AI client]
    Runtime --> DatabaseReady{Database connection works?}
    DatabaseReady -- No --> DatabaseError[Show database connection message]
    DatabaseReady -- Yes --> Conversation[Create or resume conversation]
    Conversation --> History[Load and render persisted messages]
    History --> Question[User submits a question]

    Question --> UserMessage[Persist user message]
    UserMessage --> Run[Create running answer run]
    Run --> CommitAudit[Commit message and audit record]
    CommitAudit --> Active{Active document version exists?}

    Active -- No --> NotReady[Mark run failed with DOCUMENT_NOT_READY]
    Active -- Yes --> QueryEmbedding[Embed the question through ScaDS.AI]
    QueryEmbedding --> VectorSearch[Search active pgvector chunks]
    VectorSearch --> Diversity[Apply section diversity and top-k limit]
    Diversity --> Threshold[Apply optional similarity threshold]
    Threshold --> PersistResults[Persist ranked retrieval results]
    PersistResults --> Evidence{Selected evidence exists?}

    Evidence -- No --> NoEvidence[Persist a no-reliable-information answer]
    Evidence -- Yes --> Prompt[Build evidence-only prompt]
    Prompt --> Generate[Generate answer through ScaDS.AI]
    Generate --> Complete[Persist assistant message, citations,<br/>tokens, models, and latency]

    NoEvidence --> Render[Render answer and sources]
    Complete --> Render
    Render --> Rerun[Streamlit reruns and displays persisted history]

    QueryEmbedding -. API failure .-> FailedRun[Mark answer run failed]
    Generate -. API failure .-> FailedRun
    VectorSearch -. internal failure .-> FailedRun
    FailedRun --> Warning[Show safe English error message]
```

### Retrieval Rules

1. Only chunks from the active document version are eligible.
2. The query uses pgvector cosine distance through the `<=>` operator.
3. Candidate ordering uses distance first and stable chunk ID second.
4. The database initially returns a larger candidate pool when section diversity
   is enabled.
5. At most three chunks per section are retained by default.
6. The final result list is truncated to `retrieval_top_k`.
7. An optional similarity threshold controls which results enter the model
   context.
8. Citations are grouped by section path and retain deterministic display order.

## 8. Persistence Map

`src/db.py` defines the complete persistence model.

```mermaid
flowchart LR
    Document[documents] --> Version[document_versions]
    Document --> Job[ingestion_jobs]
    Version --> Job
    Version --> Chunk[chunks with vectors]

    User[users] --> Conversation[conversations]
    Conversation --> Message[messages]
    Message --> Run[answer_runs]
    Version --> Run
    Run --> Result[retrieval_results]
    Chunk --> Result
    Run --> Assistant[assistant message]
    Assistant --> Citation[message_citations]
    Result --> Citation

    Message --> Feedback[feedback]
    User --> Feedback
    Conversation --> Summary[conversation_summaries]
    Message -. future memory source .-> HistoryVector[history_embeddings]
    Summary -. future memory source .-> HistoryVector
```

### Main Record Lifecycles

- `document_versions`: `building` to `active`, `superseded`, or `failed`.
- `ingestion_jobs`: `pending` or `running` to `succeeded` or `failed`.
- `answer_runs`: `running` to `succeeded` or `failed`.
- `messages`: immutable ordered user and assistant conversation entries.
- `retrieval_results`: ranked evidence candidates recorded for each answer run.
- `message_citations`: links displayed assistant answers to selected retrieval
  results.

## 9. External Boundaries

```mermaid
flowchart LR
    Application[TRR 339 RDM Assistant]
    Application -->|SQLAlchemy and psycopg| Postgres[(PostgreSQL and pgvector)]
    Application -->|OpenAI-compatible HTTPS| Embeddings[ScaDS.AI embeddings endpoint]
    Application -->|OpenAI-compatible HTTPS| Chat[ScaDS.AI chat endpoint]
    Application -->|Local file read| Manual[data/Manual-RDM.md]
    Browser[Web browser] -->|Streamlit session| Application
```

The application does not use the OpenAI public endpoint unless `SCADS_API_URL`
is explicitly configured that way. The ScaDS.AI base URL and key determine the
actual external service boundary.

