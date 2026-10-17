# MLQuest Development Handoff

## Goal

Continue building MLQuest, a local CPU-first, tabular ML competition research agent. The v0 target is a reproducible, budget-aware workflow that accepts a local challenge description and dataset, analyses the data, establishes baselines, performs online research, runs controlled experiments, records evidence, verifies a final solution, and produces a complete solution guide.

The next implementation goal is Step 8: build the Docker experiment runner using the completed evaluation and budget/resource layers.

## Context files to read first

Read these files in this order before continuing:

1. `AGENTS.md` — repository rules, Python environment, testing, front-end, and git instructions.
2. `PROJECT_CONTEXT.md` — product vision, confirmed scope, budgets, autonomy modes, data policy, reproducibility, and deliverables.
3. `README.md` — current project overview, intended CLI, task configuration, setup, and operating constraints.
4. `V0_IMPLEMENTATION_PLAN.md` — checklist-based implementation plan and acceptance criteria.
5. `PROGRESS_TLDR.md` — concise summary of completed phases and current next phase.

For implementation details, inspect:

- `src/mlquest/domain/` — typed domain contracts.
- `src/mlquest/persistence/` — SQLite ledger and JSON record persistence.
- `src/mlquest/artifacts/` — immutable SHA-256 content-addressed artifact store.
- `src/mlquest/datasets/` — CSV/Parquet registration, bounded profiling, warnings, reports, and lineage helpers.
- `src/mlquest/sessions/` — durable project/session service.
- `src/mlquest/cli.py` — Typer CLI commands and Rich status output.
- `tests/unit/domain/` — domain contract tests.
- `tests/unit/persistence/` — SQLite ledger tests.
- `tests/unit/artifacts/` — artifact-store tests.
- `tests/unit/datasets/` — dataset ingestion/profiling tests.
- `tests/unit/cli/` — CLI lifecycle tests.

## Current repository state

- Branch: `master`
- Latest project commit: `085e609 feat: add CLI session lifecycle`
- Current implementation progress: **8 of 16 phases complete, 50%**.
- The latest implementation commit is `085e609`; Step 6 and Step 7 implementation and documentation changes are currently uncommitted.
- The following generated tool directories remain intentionally untracked and should not be included in project commits:
  - `.agents/`
  - `.codex/`
  - `.impeccable/`
  - `.opencode/`

## Completed work

### Step 0 — Product contract and repository rules

- Defined the initial competition-solving product scope.
- Chose tabular classification/regression as the v0 domain.
- Chose local CPU execution, Docker isolation, OpenRouter, online research, CLI-first interaction, and a future localhost UI.
- Defined approximately 120 minutes for a session and 90 minutes for cumulative training/inference by default.
- Defined immutable source data, derived-data lineage, evidence-backed improvements, manual external score recording, and no automatic competition submission.
- Added `AGENTS.md`, `README.md`, `PROJECT_CONTEXT.md`, and the checklist implementation plan.

### Step 1 — Reproducible project bootstrap

- Pinned Python 3.12.x in `.python-version`.
- Added `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `Dockerfile`, and `Makefile`.
- Added the initial `mlquest` package and CLI entry point.
- Verified dependency sync, CLI help, tests, and Docker image build.

### Step 2 — Domain schemas and artifact contracts

- Added Pydantic models for tasks, datasets, artifacts, lineage, budgets, resources, phases, approvals, validation, experiments, runs, metrics, research, evidence, verification, and external scores.
- Added stable identifiers, schema versions, timezone-aware timestamps, phase-transition rules, and comparison-context identity.

### Step 3 — SQLite persistence and artifact store

- Added a versioned SQLite ledger with WAL mode and migrations.
- Added durable sessions, phases, approvals, records, events, and resume queries.
- Added deterministic JSON serialization and record rehydration.
- Added immutable SHA-256 content-addressed source and derived artifact storage.
- Added lineage manifests, safe path handling, integrity checks, overwrite protection, and artifact registration metadata.

### Step 4 — CLI shell and session lifecycle

- Added `init`, `start`, `status`, `approve`, `reject`, `pause`, and `resume`.
- Added experiment and research inspection commands.
- Added external score recording with validation and provenance.
- Added explicit `verify`, `report`, `package`, and `ui` command surfaces that clearly report unavailable later services.
- Added durable `SessionService` that reopens SQLite state for each command.
- Added Rich tables for status and pending approvals.

### Step 5 — Dataset ingestion, profiling, and lineage

- Added CSV and Parquet source registration.
- Added bounded deterministic profiling and sampling metadata.
- Added schema, dtype, row-count, missingness, cardinality, duplicate, constant-column, and target-distribution analysis.
- Added conservative target, ID, group, and time suggestions.
- Added leakage and validation warnings.
- Added train/test schema comparison and cross-split duplicate detection.
- Added machine-readable profiles, Markdown reports, and derived-data lineage helpers.

### Step 6 — Metrics, splits, validation, and baselines

- Added an explicit classification/regression metric registry with fixed direction and prediction contracts.
- Added deterministic random, stratified, group-aware, time-aware, holdout, and cross-validation split plans with serialized assignments and SHA-256 identities.
- Added prediction shape, numeric type, finite-value, probability, and label-domain validation.
- Added fold-safe trivial, scikit-learn, and LightGBM CPU baselines with fold-local preprocessing.
- Added fold-level and aggregate metric results with standard deviation and runtime/resource metadata.

### Step 7 — Budget and resource management

- Added `ResourceDetector` for CPU, memory, disk, optional GPU, Python, and Docker capability snapshots.
- Extended resource contracts with available memory and free disk measurements.
- Added `BudgetManager`, cost estimation, persisted snapshots, monotonic/deadline accounting, preflight admission, reserve protection, and actual outcome charging.
- Added bounded subprocess execution with timeout, process-group termination, and output quotas.
- Integrated resource snapshots, budget snapshots, and live remaining budget fields into `SessionService` and CLI status output.

## Verification evidence

### Tests — passed

The requested Conda command was attempted from `/home/raj/Codec/mlquest`, but the current container does not provide a usable `conda` executable. The equivalent project-local Python 3.12 environment was used for verification:

```bash
.venv/bin/python -m pytest
```

Observed result at the latest verified revision:

```text
68 passed in 7.97s
```

The suite covers domain, artifact, persistence, dataset, CLI, evaluation, budgets, and resource detection behavior. The focused Step 7 command passed with 21 tests, and focused Ruff checks passed for the new budget/resource packages.

### Docker build — passed previously

```bash
docker build -t mlquest:v0-bootstrap .
```

The image was successfully built and tagged as `mlquest:v0-bootstrap`. Re-run if the current branch changes Docker-related files.

### Diff validation — passed

```bash
git diff --check
```

## Incomplete work

The following are not implemented yet:

- Step 8: application-integrated Docker experiment runner.
- Step 9: OpenRouter provider adapter and structured planning responses.
- Step 10: fastCRW research gateway and source provenance workflow.
- Step 11: planner, scheduler, Optuna search, and adaptive experiment loop.
- Step 12: candidate confirmation, finalisation, and clean verification.
- Step 13: evidence-driven Markdown/HTML report generation.
- Step 14: localhost visual interface.
- Step 15: full offline end-to-end fixture and final hardening.

The CLI command surfaces `verify`, `report`, `package`, and `ui` exist, but intentionally report that their underlying services are not implemented yet.

## Next step: Step 8

Start by reading the Step 8 section in `V0_IMPLEMENTATION_PLAN.md` and the existing execution contracts in:

```text
src/mlquest/budgets/
src/mlquest/evaluation/
src/mlquest/domain/experiments.py
src/mlquest/persistence/
```

Implement the Docker experiment runner and connect it to `BudgetManager.run_limited_command`, preserving authoritative charging and durable session state.

Keep internal validation as the authoritative experiment-comparison signal. Do not use external leaderboard observations as a substitute for local evidence. Keep preprocessing inside validation folds and preserve comparison-context identity.

After implementation, run the project test command with the UV-managed `.venv`, update the checklist and progress summary with actual results, inspect the diff, and ask for approval before committing unless the user explicitly authorizes the commit.

## Important constraints

- Use the project-local UV-managed `.venv` and the committed `uv.lock` for all Python commands.
- Keep code clean and comments minimal; comments should explain only specific approaches or non-obvious constraints.
- Use PNPM for any future web front-end work.
- Use the `impeccable` skill for interface design or polishing.
- Add focused unit tests for each implementation phase.
- Never place secrets, credentials, local datasets, generated caches, or tool directories in commits.
- Do not claim a test, build, lint, typecheck, or verification result unless it was actually run.
