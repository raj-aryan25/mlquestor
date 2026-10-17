# MLQuest Development Handoff

## Goal

Continue building MLQuest, a local CPU-first, tabular ML competition research agent. The v0 target is a reproducible, budget-aware workflow that accepts a local challenge description and dataset, analyses the data, establishes baselines, performs online research, runs controlled experiments, records evidence, verifies a final solution, and produces a complete solution guide.

The next implementation goal is Step 11: build the planner, scheduler, and adaptive experiment loop using the completed evaluation, budget/resource, Docker runner, advisory provider, and research layers.

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
- `src/mlquest/providers/` — OpenRouter advisory provider contracts and adapter.
- `src/mlquest/research/` — provider-neutral research protocol, fastCRW adapter, safety, cache, and service.
- `tests/unit/domain/` — domain contract tests.
- `tests/unit/persistence/` — SQLite ledger tests.
- `tests/unit/artifacts/` — artifact-store tests.
- `tests/unit/datasets/` — dataset ingestion/profiling tests.
- `tests/unit/cli/` — CLI lifecycle tests.
- `tests/unit/runner/` — isolated Docker runner tests using a fake Docker executable.
- `tests/unit/providers/` — offline OpenRouter adapter, redaction, and authority tests.
- `tests/unit/research/` — offline fastCRW, URL safety, caching, and provenance tests.

## Current repository state

- Branch: `main`
- Latest project commit: `f22eb61 feat: add OpenRouter and fastCRW adapters`
- Current implementation progress: **11 of 16 phases complete, 68.75%**.
- Step 8 implementation and documentation changes are committed in `9d55b69`.
- Step 9 and Step 10 implementation and documentation changes are committed in `f22eb61`.
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

### Step 8 — Docker experiment runner

- Added an isolated Docker command builder and `DockerExperimentRunner`.
- Added `docker/experiment/Dockerfile` for the non-root experiment image using the pinned project lockfile.
- Enforced read-only root/source mounts, no network, non-root execution, dropped capabilities, no-new-privileges, CPU/memory/PID limits, and bounded output/time.
- Kept the SQLite ledger and artifact store outside the container and persisted run attempts and captured artifacts after execution.
- Integrated one-run execution into `SessionService.run_docker_experiment`.

### Step 9 — OpenRouter provider adapter

- Added provider-neutral structured request/result contracts and typed advisory planning schemas.
- Added an OpenAI-compatible OpenRouter adapter with configurable model, fallback models, token/timeout/prompt limits, temperature, and bounded retry/backoff policy.
- Added local Pydantic validation for task interpretations, validation plans, research queries, hypotheses, experiment proposals, conclusions, and report sections.
- Added prompt redaction and size limits; API keys are read from the environment and never persisted or included in request JSON.
- Added SQLite schema version 4 with typed `ProviderCallRecord` metadata for provider/model, usage, digests, attempts, status, and failure code without request or response bodies.
- Added offline provider, persistence, retry, malformed-output, authority-boundary, and redaction tests.

### Step 10 — fastCRW research gateway

- Added provider-neutral `ResearchProvider` contracts for bounded search and page-fetch operations.
- Added a separately deployed fastCRW HTTP adapter with local-by-default endpoint policy, explicit host allowlists, retries, rate limits, response-size bounds, media-type checks, and redirect controls.
- Added URL validation and SSRF protections for credentials, private hosts, unsafe schemes, HTTPS downgrade, and malformed redirects.
- Added immutable content-addressed research snapshots with raw and normalized content hashes.
- Added provenance-backed `ResearchService` records, cache artifact links, source events, failure events, and untrusted-content safety flags.
- Added offline tests for search/fetch decoding, retries, malformed responses, size/media limits, URL safety, caching, and ledger provenance.

## Verification evidence

### Tests — passed

The project-local UV-managed Python 3.12 environment was used for verification:

```bash
make test
```

Observed result at the latest verified revision:

```text
90 passed
```

The suite covers domain, artifact, persistence, dataset, CLI, evaluation, budgets, resource detection, Docker runner, provider, and research behavior. The focused provider command passed with 8 tests and the focused research command passed with 7 tests. Ruff and mypy also passed for the complete source tree.

No live fastCRW service is required for verification; the research adapter is tested through injected offline HTTP transports. The adapter expects fastCRW to remain a separately deployed service pending licensing and deployment review.

### Docker build — passed on the developer host

```bash
docker build -t mlquest:dev .
docker build -f docker/experiment/Dockerfile -t mlquest-experiment:dev .
```

Both application and experiment images were successfully built on the developer host. The repository verification container did not expose Docker, so real daemon-backed fixture execution was not run by the agent; the runner's offline tests use an injectable fake Docker executable.

### Diff validation — passed

```bash
git diff --check
```

## Incomplete work

The following are not implemented yet:

- Step 11: planner, scheduler, Optuna search, and adaptive experiment loop.
- Step 12: candidate confirmation, finalisation, and clean verification.
- Step 13: evidence-driven Markdown/HTML report generation.
- Step 14: localhost visual interface.
- Step 15: full offline end-to-end fixture and final hardening.

The CLI command surfaces `verify`, `report`, `package`, and `ui` exist, but intentionally report that their underlying services are not implemented yet.

## Next step: Step 11

Start by reading the Step 11 section in `V0_IMPLEMENTATION_PLAN.md` and the existing research/provider/domain contracts in:

```text
src/mlquest/domain/
src/mlquest/persistence/
src/mlquest/runner/
src/mlquest/providers/
src/mlquest/research/
```

Implement the planner and experiment loop with declarative, validated proposals, approval/budget gates, and Docker-only execution. Keep research text untrusted, keep the model advisory, and preserve internal validation as the authoritative comparison signal.

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
