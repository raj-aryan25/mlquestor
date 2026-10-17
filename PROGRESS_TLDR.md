# MLQuest Progress TL;DR

## Current state

MLQuest is a local, CPU-first, tabular ML competition research agent. The project is currently at **9 of 16 implementation phases complete (56.25%)**.

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

### Step 6 — Metrics, splits, validation, and baselines

- Added an explicit classification and regression metric registry with direction-aware comparison.
- Added deterministic random, stratified, group-aware, time-aware, holdout, and cross-validation split plans with persisted assignments and identities.
- Added prediction shape, type, finite-value, probability, and label-domain validation.
- Added fold-safe trivial, scikit-learn, and LightGBM CPU baselines with pipeline preprocessing.
- Added fold-level and aggregate metrics, standard deviation, runtime, CPU, memory, and thread metadata.

### Step 7 — Budget and resource management

- Added side-effect-free CPU, memory, disk, GPU, Python, and Docker capability detection.
- Added monotonic elapsed-time tracking with persisted wall-clock deadlines and resumable snapshots.
- Added separate session and model-execution accounting with a protected finalisation reserve.
- Added explainable cost estimation using model family, dataset size, configuration, and prior observations.
- Added preflight admission, max-experiment/resource-limit checks, actual charging for success/failure/timeout/cancellation, and bounded subprocess execution.
- Integrated resource profiles and live remaining budgets into durable sessions and the CLI status view.

### Step 8 — Docker experiment runner

- Added an application-integrated Docker runner with read-only root and source mounts.
- Added a pinned-dependency experiment image definition at `docker/experiment/Dockerfile`.
- Disabled experiment-container networking and dropped all Linux capabilities.
- Added non-root execution, no-new-privileges, CPU, memory, PID, wall-clock, and output limits.
- Kept SQLite ledgers and artifact manifests outside the container boundary.
- Added durable run-attempt records, captured logs/output artifacts, cleanup, and budget charging.
- Integrated one-run execution through `SessionService.run_docker_experiment`.

## Verification status

The full offline test suite passes in the project-local UV-managed Python 3.12 environment:

```text
73 passed in 21.23s
```

The Docker image was built successfully as:

```text
mlquest:v0-bootstrap (the current container does not provide a Docker executable for a fresh rebuild)
```

## Not implemented yet

- OpenRouter provider adapter.
- fastCRW research adapter.
- Experiment planner and autonomous search loop.
- Confirmation and final-solution verification workflow.
- Report generation.
- Localhost visual interface.


## Next phase

**Step 9 — Implement the OpenRouter LLM adapter.**

The Docker runner now uses the Step 7 admission and bounded-execution contracts to run approved ML experiments inside Docker with filesystem, network, process, CPU, memory, and wall-clock controls. The next phase is the provider adapter for structured planning responses.
