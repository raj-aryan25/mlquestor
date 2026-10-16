# MLQuest Progress TL;DR

## Current state

MLQuest is a local, CPU-first, tabular ML competition research agent. The project is currently at **4 of 16 implementation phases complete (25%)**.

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

## Verification status

The focused test suite passes in the `mlquestor` Conda environment:

```text
24 passed in 0.39s
```

The Docker image was built successfully as:

```text
mlquest:v0-bootstrap
```

## Not implemented yet

- Real session-oriented CLI commands such as `init`, `start`, `status`, and `resume`.
- Dataset ingestion and profiling.
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

**Step 4 — Implement the CLI shell and session lifecycle.**

The next phase will connect the existing domain models and persistence layer to a usable agent-style CLI. It should support project/session initialization, session start, status inspection, approval decisions, pause/resume, experiment inspection, external-score recording, verification, reporting, and packaging command surfaces.

The CLI must read and write through the persistence services rather than keeping state in process memory.
