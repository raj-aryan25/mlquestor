# MLQuest Progress TL;DR

## Current state

MLQuest is a local, CPU-first, tabular ML competition research agent. The project is currently at **6 of 16 implementation phases complete (approximately 38%)**.

## Completed phases

### Step 0 — Product contract and repository rules

- Defined the v0 product scope and competition-solving workflow.
- Documented OpenRouter, fastCRW, Docker, CLI, localhost UI, budget, reproducibility, and score-provenance decisions.
- Added repository and agent instructions in `AGENTS.md`.
- Added the project README and implementation plan.

### Step 1 — Reproducible project bootstrap

- Pinned Python 3.12.x.
- Added `pyproject.toml` and `uv.lock`.
- Added `.env.example`, `.gitignore`, Dockerfile, and Makefile.
- Added the initial `mlquest` package and CLI entry point.
- Verified `uv sync --frozen`.
- Verified the CLI help command.
- Verified the Docker image build.

### Step 2 — Domain schemas and artifact contracts

- Added Pydantic models for tasks, datasets, artifacts, budgets, resources, phases, approvals, validation, experiments, runs, metrics, research, evidence, verification, and external scores.
- Added comparison-context identity generation.
- Added valid phase-transition rules.
- Added focused domain tests.

### Step 3 — SQLite persistence and artifact store

- Added a versioned SQLite ledger with WAL mode.
- Added durable session, phase, record, event, and resume queries.
- Added JSON serialization and domain-record rehydration.
- Added append-only event storage.
- Added a SHA-256 content-addressed filesystem artifact store.
- Added immutable source artifact storage.
- Added derived artifact lineage, transformation, and fit-scope manifests.
- Added path traversal, overwrite, symlink, and integrity protections.
- Added artifact-to-ledger registration metadata.

### Step 5 — Dataset ingestion, profiling, and lineage

- Added CSV and Parquet source registration.
- Added bounded, deterministic profiling with sampling metadata.
- Added schema, dtype, row count, missingness, cardinality, duplicate, constant-column, and target-distribution analysis.
- Added conservative target, ID, group, and time-column suggestions.
- Added leakage and validation warnings.
- Added train/test schema comparison and cross-split duplicate detection.
- Added machine-readable profile models and Markdown reports without raw data rows.
- Added derived-data storage helpers with parent artifact IDs, transformation descriptions, and fit scope.

### Step 4 — CLI shell and session lifecycle

- Added the `init`, `start`, `status`, `approve`, `reject`, `pause`, and `resume` commands.
- Added experiment and research inspection command surfaces.
- Added external score recording with validation and provenance.
- Added explicit `verify`, `report`, `package`, and `ui` command surfaces that report unavailable later services clearly.
- Added Rich status tables and durable session reopening through SQLite.
- Added CLI lifecycle tests covering initialization, status readback, approvals, pause/resume, and external scores.

## Verification status

The verified test suite passes in the `mlquestor` Conda environment:

```text
29 passed in 0.69s
```

The Docker image was built successfully as:

```text
mlquest:v0-bootstrap
```

## Not implemented yet

- Metric registry and baseline training.
- Budget manager and resource detector.
- Application-integrated Docker experiment runner.
- OpenRouter provider adapter.
- fastCRW research adapter.
- Experiment planner and autonomous search loop.
- Confirmation and final-solution verification workflow.
- Report generation.
- Localhost visual interface.


## Next phase

**Step 6 — Implement metrics, splits, validation, and baselines.**

The next phase will establish the authoritative evaluation layer: metric registration, split strategies, baseline models, prediction validation, and reproducible score comparison.
