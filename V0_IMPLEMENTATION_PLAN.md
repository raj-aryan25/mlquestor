# MLQuest v0 Implementation Plan

## 1. Purpose

This document defines the step-by-step implementation plan for the first usable version of the autonomous ML competition research agent described in `PROJECT_CONTEXT.md`.

The plan intentionally starts with a reliable vertical slice rather than a broad autonomous system. The first release must be able to take a local tabular competition task, analyse it, establish a baseline, perform online research, run bounded experiments, record evidence, verify a final solution, and produce a reproducible solution package.

No application code is specified or implemented by this document. It is the implementation sequence and acceptance plan to use when development begins.

## Implementation progress

Progress is tracked at the phase level and in the task and acceptance checklists below.

- [x] **Step 0 — Freeze the product contract and repository rules** — complete in the working tree; README and product-contract decisions are documented.
- [x] **Step 1 — Bootstrap a reproducible Python project** — complete.
- [x] **Step 2 — Define the domain schema and artifact model** — implementation and focused tests complete.
- [ ] **Step 3 — Build SQLite persistence and the artifact store** — not started.
- [ ] **Step 4 — Implement the CLI shell and session lifecycle** — not started.
- [ ] **Step 5 — Implement dataset ingestion, profiling, and lineage** — not started.
- [ ] **Step 6 — Implement metrics, splits, validation, and baselines** — not started.
- [ ] **Step 7 — Implement budget and resource management** — not started.
- [ ] **Step 8 — Build the Docker experiment runner** — not started.
- [ ] **Step 9 — Implement the OpenRouter LLM adapter** — not started.
- [ ] **Step 10 — Implement the research gateway with fastCRW** — not started.
- [ ] **Step 11 — Implement the planner, scheduler, and experiment loop** — not started.
- [ ] **Step 12 — Implement confirmation, finalisation, and verification** — not started.
- [ ] **Step 13 — Implement evidence-driven report generation** — not started.
- [ ] **Step 14 — Add the localhost visual interface** — not started.
- [ ] **Step 15 — Build the end-to-end fixture and project checks** — not started.

**Current phase count:** 3 of 16 checklist phases complete, or approximately 19% by phase count. Steps 0, 1, and 2 are complete.

## 2. v0 product contract

### 2.1 Supported problem class

- Tabular supervised classification and regression.
- CSV and Parquet input as the initial required formats.
- Optional sample submission and data dictionary.
- User-provided primary metric and metric direction.
- Local machine execution with CPU as the required target.
- Optional GPU detection only; GPU-specific training is not a v0 requirement.

### 2.2 User experience

The primary interface is an agent-style CLI that can be started from a project directory and can access local files explicitly supplied to a session.

The CLI should support both:

- autonomous mode, in which approved workflow phases proceed automatically; and
- approval-gated mode, in which the system pauses at phase boundaries and presents the proposed next action, estimated cost, risks, and remaining budget.

A local visual interface should expose the same session state, experiment history, approvals, logs, reports, and actions as the CLI. It must use the same application services and ledger; it must not become a second orchestration implementation.

### 2.3 Online and provider policy

- v0 requires online access for LLM reasoning and web research.
- OpenRouter is the first LLM provider integration.
- Free OpenRouter models should be usable when available, but the model name must be configurable and no free-model availability should be assumed.
- API keys belong in environment variables or a local ignored `.env` file; never in task files, prompts, logs, reports, or git.
- Research and LLM providers receive task summaries and bounded metadata by default, not raw dataset contents.

### 2.4 Execution policy

- Docker is the v0 experiment boundary.
- Experiment containers run with no general outbound network access.
- Source datasets are mounted read-only.
- Run-specific work and output directories are writable only where required.
- Generated code cannot write the source dataset or the authoritative ledger directly.
- The runner records authoritative exit status, resource use, output validation, and failure state.

### 2.5 Budget policy

The default budgets are:

- approximately 120 minutes for the complete session;
- approximately 90 minutes for cumulative model training and inference.

Both values are configurable. The scheduler must reserve time for confirmation, final training, verification, packaging, and reporting before admitting exploratory experiments.

### 2.6 Score and submission policy

The v0 evaluation model has three score sources:

1. **Internal validation:** the default and authoritative score for local experiment comparison.
2. **User-provided labelled holdout/test data:** evaluated locally when the user supplies labels and explicitly identifies the split.
3. **User-reported external score:** manually entered after the user evaluates or submits the generated artifact elsewhere.

External scores are recorded as external observations with provenance. They are not silently treated as reproducible local evidence. v0 generates a submission file or solution artifact but does not upload it automatically to Kaggle or another platform.

## 3. Recommended v0 technology baseline

- Python 3.12.x.
- `pyproject.toml` and a committed `uv.lock`.
- Pydantic v2 for domain models and structured LLM responses.
- Explicit application-owned phase state machine.
- SQLite for sessions, phases, approvals, experiments, metrics, and evidence.
- Content-addressed or immutable filesystem artifacts.
- Typer and Rich for the CLI.
- FastAPI plus a lightweight server-rendered or minimal browser UI for localhost visibility and approvals.
- pandas and PyArrow for the model boundary and artifacts.
- DuckDB or equivalent scan/query support for resource-aware large-data profiling.
- Pandera plus project-specific data-quality and leakage checks.
- scikit-learn for preprocessing, metrics, and baselines.
- LightGBM as the initial strong CPU tabular model.
- Optuna for bounded experiment search.
- Jinja2 for Markdown and HTML reports.
- Docker for the experiment runner.
- `pytest`, Hypothesis, Ruff, and a type checker as project checks.
- fastCRW/crw behind a research-provider interface for the initial search/scrape backend.

The application should not depend directly on a heavyweight agent framework in v0. The LLM layer should be a small provider adapter with typed requests, typed responses, bounded tool calls, and explicit orchestration outside the model.

## 4. Implementation principles

1. Build the evidence loop before making the agent highly autonomous.
2. Keep the LLM advisory and proposal-oriented; keep execution and evaluation deterministic.
3. Preserve original data and make every derived artifact traceable.
4. Make every phase resumable from durable state.
5. Enforce budgets before starting work, not only after it finishes.
6. Treat web content and model outputs as untrusted input.
7. Generate final reports from persisted records, not conversation memory.
8. Keep external research replaceable and isolated from ML execution.
9. Make the offline test suite independent of OpenRouter, CRW, and any external API.
10. Keep the first model portfolio small enough that each result can be understood and reproduced.

## 5. Step-by-step implementation sequence

### Step 0 — Freeze the product contract and repository rules — Complete

**Goal:** Make the first implementation unambiguous before writing application code.

**Tasks:**

- [x] Confirm this plan and the decisions in `PROJECT_CONTEXT.md`.
- [x] Define v0 terminology: session, phase, task specification, source artifact, derived artifact, experiment, run, evidence claim, verification run, and external score.
- [x] Define the initial command vocabulary and session directory convention.
- [x] Write the project README with setup, run, test, and local deployment expectations.
- [x] Decide the initial supported metric registry and explicitly reject unsupported metrics instead of guessing.
- [x] Document the Docker prerequisite and the online-only LLM/research requirement.

**Status:** Complete in the working tree. The README skeleton and product decisions are documented; the Step 0 documentation changes still require their own commit.

**Deliverable:** agreed product contract and an updated README skeleton.

**Acceptance checks:**

- [x] A new contributor can identify the first user flow from the README.
- [x] The README states that no automatic competition submission is performed.
- [x] Secrets, source data, and generated artifacts have documented storage boundaries.

### Step 1 — Bootstrap a reproducible Python project — Complete

**Goal:** Create a clean, pinned, testable foundation.

**Tasks:**

- [x] Pin Python 3.12 in `.python-version`.
- [x] Create `pyproject.toml` with runtime and development dependencies.
- [x] Generate `uv.lock`.
- [x] Create `.env.example` with placeholders for OpenRouter and research-provider configuration.
- [x] Add `.gitignore` for secrets, local datasets, session artifacts, caches, and generated reports where appropriate.
- [x] Add a project `Dockerfile` that installs the application and all system dependencies required for a clean build and run.
- [x] Add a `Makefile` with one offline `make test` command and separate local quality/build commands as useful.
- [x] Add the initial package layout without business logic.

**Deliverable:** the project installs and runs a placeholder CLI in a clean Linux environment.

**Status:** Bootstrap files are present. `uv sync --frozen`, the CLI smoke check, the test suite, and the Docker image build succeeded.

**Acceptance checks:**

- [x] `uv sync --frozen` succeeds.
- [x] `make test` succeeds without network access or provider credentials.
- [x] `docker build` succeeds using only project files and pinned dependencies.
- [x] No global Python package installation is required.

### Step 2 — Define the domain schema and artifact model — Complete

**Goal:** Establish the contracts that all later components use.

**Tasks:**

- [x] Define Pydantic models for:
  - [x] task specification;
  - [x] dataset roles and source artifacts;
  - [x] resource profile;
  - [x] budget configuration;
  - [x] phase state;
  - [x] approval request and decision;
  - [x] validation and split specification;
  - [x] experiment specification;
  - [x] run attempt;
  - [x] metric result;
  - [x] research record;
  - [x] evidence claim;
  - [x] verification result;
  - [x] external score observation.
- [x] Define stable IDs, timestamps, status enums, and schema version fields.
- [x] Define artifact manifests with content hashes, media types, parent references, and immutable locations.
- [x] Define valid status transitions and reject invalid transitions.
- [x] Define the comparison context used to determine whether two scores are comparable.

**Status:** Domain contracts and focused tests are present, and the focused test suite passes in the `mlquestor` environment.

**Deliverable:** versioned domain contracts independent of CLI, web, LLM, and database implementations.

**Acceptance checks:**

- [x] Invalid task specs and experiment specs fail validation.
- [x] Phase transitions reject illegal jumps.
- [x] A comparison context changes when the dataset, split, metric, or target definition changes.
- [x] Model serialization of all persisted records is deterministic.

### Step 3 — Build SQLite persistence and the artifact store — Not started

**Goal:** Make sessions, checkpoints, evidence, and files durable.

**Tasks:**

- [ ] Create SQLite tables for sessions, phases, approvals, artifacts, datasets, derived-data lineage, research records, hypotheses, experiments, run attempts, metrics, claims, and verification runs.
- [ ] Add schema versioning and migration support.
- [ ] Store large files in an immutable artifact directory rather than SQLite blobs.
- [ ] Add content hashing and manifest writing.
- [ ] Add append-only event records for important transitions.
- [ ] Add safe path handling to prevent artifact path traversal.
- [ ] Add resume queries for incomplete phases and in-flight runs.

**Deliverable:** a session can be created, checkpointed, closed, and resumed without relying on process memory.

**Acceptance checks:**

- [ ] Restarting a process preserves session and phase state.
- [ ] Duplicate artifact content resolves to a stable identity.
- [ ] Original source artifacts cannot be overwritten through application services.
- [ ] Interrupted runs are not automatically marked successful.

### Step 4 — Implement the CLI shell and session lifecycle — Not started

**Goal:** Provide the agent-style local entry point before adding intelligence.

**Tasks:**

- [ ] Implement `mlquest --help` and a clear command hierarchy.
- [ ] Add commands for:
  - `init` — create a local MLQuest project/session template;
  - `start` — create a session from a task file and local dataset paths;
  - `status` — show current phase, best candidate, remaining budgets, and pending approvals;
  - `approve` / `reject` — resolve a phase approval request;
  - `pause` / `resume` — control a session safely;
  - `experiments` — list and inspect runs;
  - `research` — inspect recorded sources and hypotheses;
  - `score add` — record a user-provided external score;
  - `verify` — run final verification;
  - `report` — generate reports;
  - `package` — create the final solution bundle.
- [ ] Add a human-readable interactive session view with Rich.
- [ ] Ensure every command reads state from the persistence layer rather than a process-global object.
- [ ] Make commands safe to run repeatedly.

**Deliverable:** a user can initialise and inspect a session, even before the agent can run experiments.

**Acceptance checks:**

- [ ] CLI help works without credentials or network access.
- [ ] A session can be paused and resumed.
- [ ] Approval decisions are persisted and visible in status output.
- [ ] External scores require source, metric, direction, timestamp, and user attribution fields.

### Step 5 — Implement dataset ingestion, profiling, and lineage — Not started

**Goal:** Analyse local tabular data without mutating source files or overloading the LLM.

**Tasks:**

- [ ] Register source files by hash and role: train, test, sample submission, metadata, or auxiliary input.
- [ ] Preserve source paths and copy or reference them in an immutable source area.
- [ ] Detect CSV and Parquet schemas, row counts, column types, missingness, cardinality, duplicates, constant columns, and target distribution.
- [ ] Use bounded scans and samples for large inputs; use DuckDB/PyArrow where full pandas loading is unnecessary.
- [ ] Identify likely target, ID, group, and time columns as suggestions requiring confirmation when ambiguous.
- [ ] Compare train and test schemas and basic distributions.
- [ ] Generate a machine-readable data profile and a human-readable report.
- [ ] Record suspected leakage and validation warnings without silently altering data.
- [ ] Define derived-data transformations as manifests with parent hashes and fit scope.

**Deliverable:** a dataset phase that produces a durable data profile and validation recommendations.

**Acceptance checks:**

- [ ] Source files remain byte-for-byte unchanged.
- [ ] Profiles are deterministic for the same input and configuration.
- [ ] Large-data profiling uses bounded memory and documents sampling.
- [ ] Raw data rows are never sent to OpenRouter or the research provider by default.
- [ ] Obvious duplicate, missing-value, schema, and target leakage warnings are visible.

### Step 6 — Implement metrics, splits, validation, and baselines — Not started

**Goal:** Establish a trustworthy local comparison context.

**Tasks:**

- [ ] Create a metric registry with explicit direction and validation rules.
- [ ] Initially support a deliberately small set of common metrics, such as:
  - accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, and log loss for classification;
  - MAE, MSE/RMSE, and R2 for regression.
- [ ] Reject unknown metrics unless the user provides a validated custom metric module under an explicit policy.
- [ ] Implement deterministic random, stratified, group-aware, and time-aware split strategies where applicable.
- [ ] Store fold assignments or split seeds and hashes.
- [ ] Build trivial baselines and simple scikit-learn baselines.
- [ ] Add one strong LightGBM baseline with controlled thread counts.
- [ ] Validate prediction shape, type, finite values, label domain, and submission schema.
- [ ] Store fold-level and aggregate metrics, runtime, and resource usage.

**Deliverable:** a baseline phase with comparable, reproducible scores.

**Acceptance checks:**

- [ ] The same task, seed, split, and environment produce scores within configured tolerance.
- [ ] Preprocessing is fitted inside the training fold through a pipeline.
- [ ] The metric direction is handled correctly when ranking candidates.
- [ ] An invalid prediction cannot become a successful experiment.
- [ ] Baseline reports clearly distinguish trivial, simple, and strong baselines.

### Step 7 — Implement budget and resource management — Not started

**Goal:** Make the two-hour and 90-minute constraints enforceable.

**Tasks:**

- [ ] Detect CPU count, memory, disk, optional GPU availability, Python version, and Docker capabilities.
- [ ] Use monotonic timers and persisted deadlines.
- [ ] Track separate total-session and model-execution budgets.
- [ ] Create a protected finalisation reserve.
- [ ] Estimate experiment cost from prior runs, model family, dataset size, and requested configuration.
- [ ] Require preflight approval before scheduling an expensive run.
- [ ] Add hard timeouts, process-group cancellation, and output quotas.
- [ ] Charge actual runtime and resource use even for failed or timed-out attempts.
- [ ] Stop scheduling when the remaining budget cannot support confirmation and verification.

**Deliverable:** a scheduler that refuses unsafe or over-budget work before execution.

**Acceptance checks:**

- [ ] A run that exceeds its timeout is terminated and recorded as failed.
- [ ] An experiment that threatens the final reserve is rejected.
- [ ] The status command reports remaining session and execution budgets.
- [ ] Resuming a session does not reset elapsed time.
- [ ] A budget-exhausted session still packages the best verified result available.

### Step 8 — Build the Docker experiment runner — Not started

**Goal:** Execute generated or selected ML code in a controlled, reproducible environment.

**Tasks:**

- [ ] Define an experiment image based on the pinned project runtime.
- [ ] Run as a non-root user with a read-only base filesystem.
- [ ] Mount source and approved derived artifacts read-only.
- [ ] Mount run-specific work/output/log directories with bounded write access.
- [ ] Disable outbound network access for experiment containers.
- [ ] Drop unnecessary capabilities and enable no-new-privileges.
- [ ] Apply CPU, memory, process-count, disk, and wall-clock limits.
- [ ] Capture stdout, stderr, exit code, environment metadata, and resource metrics.
- [ ] Ensure the experiment cannot access the SQLite database or rewrite authoritative manifests.
- [ ] Add a trusted-code fallback only if clearly marked as less isolated and explicitly enabled.

**Deliverable:** a reproducible, isolated runner that returns a structured run result.

**Acceptance checks:**

- [ ] A fixture experiment can read approved data and write approved outputs.
- [ ] A fixture experiment cannot modify source data.
- [ ] Network access is unavailable inside the runner.
- [ ] Timeout kills the complete process group.
- [ ] Re-running the same fixture produces equivalent artifacts and metrics.

### Step 9 — Implement the OpenRouter LLM adapter — Not started

**Goal:** Add online LLM reasoning without handing the model execution authority.

**Tasks:**

- [ ] Implement a provider protocol and OpenRouter adapter using an OpenAI-compatible API surface where appropriate.
- [ ] Read the API key from the environment only.
- [ ] Make model, temperature, token limit, timeout, and retry policy configurable.
- [ ] Add structured Pydantic response schemas for:
  - task interpretation;
  - validation-plan proposal;
  - research query;
  - hypothesis;
  - experiment specification;
  - experiment conclusion;
  - report draft sections.
- [ ] Include prompt and response metadata, model ID, and usage data in the research/planning ledger without storing secrets.
- [ ] Add bounded retries and explicit provider failure states.
- [ ] Add provider/model fallback configuration, but do not assume free models are permanently available.
- [ ] Redact or summarise sensitive dataset information before sending it to the provider.

**Deliverable:** a provider-agnostic planner interface with OpenRouter as the v0 backend.

**Acceptance checks:**

- [ ] LLM calls are mocked in offline tests.
- [ ] Malformed model output is rejected and does not reach the executor.
- [ ] The model cannot directly execute shell commands or write ledger records.
- [ ] Prompts do not contain API keys or raw dataset rows by default.
- [ ] The provider/model used for a decision is recorded.

### Step 10 — Implement the research gateway with fastCRW — Not started

**Goal:** Enable online, source-backed research while isolating untrusted web content.

**Tasks:**

- [ ] Define a `ResearchProvider` interface with search and page-fetch operations.
- [ ] Add a fastCRW adapter using its CLI, HTTP, or MCP-compatible surface based on the chosen local deployment mode.
- [ ] Prefer local/self-hosted fastCRW where practical to reduce dependence on paid search APIs.
- [ ] Keep fastCRW outside the training container and behind timeouts, response-size limits, and rate limits.
- [ ] Store source title, URL, source type, retrieval time, content hash, excerpts, and extracted findings.
- [ ] Link findings to hypotheses and proposed experiments.
- [ ] Treat all retrieved page text as untrusted content and protect the planner from prompt injection.
- [ ] Do not download or execute research code automatically.
- [ ] Review the fastCRW AGPL-3.0 engine license and SDK licenses before bundling, modifying, or redistributing it. Prefer using it as a separate service or documented prerequisite until this is resolved.
- [ ] Keep a provider interface so a different search backend can replace fastCRW later.

**Deliverable:** a research phase that produces cited, testable hypotheses.

**Acceptance checks:**

- [ ] Research failure does not corrupt or block completed local experiments.
- [ ] Sources are cached and traceable.
- [ ] Retrieved content cannot change system policy or execute code.
- [ ] Research records distinguish source claims from measured local results.
- [ ] The online integration is excluded from offline `make test`.

### Step 11 — Implement the planner, scheduler, and experiment loop — Not started

**Goal:** Close the autonomous search loop.

**Tasks:**

- [ ] Implement deterministic phase transitions and approval gates.
- [ ] Combine data findings, baseline results, research hypotheses, past experiments, failures, and remaining budget into planner context.
- [ ] Ask the LLM to propose declarative experiment specs, not shell commands.
- [ ] Validate experiment specs against an allowlisted model registry, metric registry, split policy, data lineage, and resource limits.
- [ ] Use Optuna for bounded hyperparameter search inside approved experiment families.
- [ ] Start with serial execution to make budget accounting and reproducibility straightforward.
- [ ] Rank next experiments by expected value, estimated cost, novelty, and risk.
- [ ] Record every experiment as exploratory, provisional, confirmed, or rejected.
- [ ] Add safe deterministic fallbacks when LLM planning fails.
- [ ] Stop when the budget, stopping rule, evidence threshold, or finalisation reserve requires it.

**Deliverable:** the complete baseline → hypothesis → experiment → evidence → next-experiment loop.

**Acceptance checks:**

- [ ] The loop can complete on a tiny synthetic classification task without human intervention.
- [ ] Approval mode pauses only at configured phase boundaries.
- [ ] Autonomous mode still records approval-equivalent transitions.
- [ ] Failed experiments create usable diagnostics and do not disappear from the ledger.
- [ ] A single score increase is not promoted to confirmed evidence without the configured evidence policy.

### Step 12 — Implement confirmation, finalisation, and verification — Not started

**Goal:** Turn the best candidate into a verified competition-ready artifact.

**Tasks:**

- [ ] Select candidates only from valid comparison contexts.
- [ ] Re-run the strongest candidate with configured repeated seeds, stronger folds, or an independent clean run.
- [ ] Record variance and evidence strength.
- [ ] Freeze the selected task spec, split spec, code revision, environment, model configuration, and preprocessing pipeline.
- [ ] Train the final solution using the allowed final execution budget.
- [ ] Generate predictions and validate sample-submission shape, IDs, columns, types, finite values, and row count.
- [ ] Generate a reproducible Python script and/or notebook that contains preprocessing and feature engineering.
- [ ] Add an explicit `score add` flow for manually entered competition or external test scores.
- [ ] Mark external scores as user-reported observations with source and timestamp.
- [ ] Do not upload or submit to competition platforms.

**Deliverable:** a final solution bundle containing code, configuration, model/predictions where appropriate, manifests, and verification output.

**Acceptance checks:**

- [ ] Verification runs from a clean workspace and recorded configuration.
- [ ] The final output is rejected if it does not match the expected submission contract.
- [ ] The final report distinguishes validation, labelled holdout, and user-reported external scores.
- [ ] A verification mismatch blocks the “verified” status.
- [ ] The bundle does not require the original source file to be modified.

### Step 13 — Implement evidence-driven report generation — Not started

**Goal:** Produce the complete solution guide from authoritative records.

**Tasks:**

- [ ] Generate Markdown and HTML reports with Jinja2.
- [ ] Include:
  - task interpretation;
  - dataset findings;
  - validation strategy and assumptions;
  - baseline results;
  - research sources and hypotheses;
  - experiment table;
  - accepted and rejected improvements;
  - runtime and budget usage;
  - confirmation evidence;
  - final model and preprocessing;
  - verification outcome;
  - submission instructions;
  - limitations and unresolved risks.
- [ ] Link every improvement claim to experiment IDs and metric records.
- [ ] Include environment, code, dataset, split, and artifact hashes.
- [ ] Clearly label insufficient evidence, failed runs, and user-reported external scores.
- [ ] Make report generation deterministic for the same ledger and template versions.

**Deliverable:** a complete solution guide that does not invent claims or rely on conversational history.

**Acceptance checks:**

- [ ] Every claimed improvement resolves to persisted evidence.
- [ ] Reports are reproducible from a fixture session.
- [ ] Failed experiments remain visible where they affect interpretation.
- [ ] Reports contain exact commands for rerunning the solution.

### Step 14 — Add the localhost visual interface — Not started

**Goal:** Provide a simple browser view without duplicating application logic.

**Tasks:**

- [ ] Add a local server command, such as `mlquest ui`, bound to localhost only by default.
- [ ] Expose session status, current phase, budget, best candidate, experiment history, evidence levels, logs, research sources, and generated reports.
- [ ] Add approval controls for phase gates.
- [ ] Add pause/resume controls with confirmation for destructive or expensive actions.
- [ ] Add a form for user-provided external score observations.
- [ ] Add links to local artifact files and reports through safe, scoped routes.
- [ ] Use the same service layer as the CLI and do not allow browser requests to bypass validation, budget, or approval policies.
- [ ] Keep the UI deliberately small in v0; visualisation and control are more important than a full experiment editor.

**Deliverable:** a localhost dashboard that observes and controls the same CLI-created session.

**Acceptance checks:**

- [ ] CLI and UI show the same persisted state.
- [ ] Approval made in the UI is visible in the CLI and vice versa.
- [ ] The UI cannot expose files outside the session artifact scope.
- [ ] The UI remains local-only unless the user explicitly configures otherwise.

### Step 15 — Build the end-to-end fixture and project checks — Not started

**Goal:** Prove the v0 vertical slice offline and document online prerequisites separately.

**Tasks:**

- [ ] Add tiny synthetic classification and regression fixtures.
- [ ] Add an offline end-to-end test covering:
  - session creation;
  - dataset registration;
  - profiling;
  - split and metric setup;
  - baseline;
  - one experiment;
  - evidence recording;
  - confirmation;
  - final packaging;
  - verification;
  - report generation.
- [ ] Mock OpenRouter and fastCRW in offline tests.
- [ ] Add tests for Docker runner policy using a small fixture image or a documented integration test path.
- [ ] Add tests for timeouts, invalid predictions, leakage warnings, budget exhaustion, resume, and external score recording.
- [ ] Ensure `make test` is the single offline test command.
- [ ] Add optional commands for lint, typecheck, and Docker build without replacing `make test`.

**Deliverable:** a repeatable, offline-tested v0 workflow plus documented online setup.

**Acceptance checks:**

- [ ] `make test` passes on a clean Linux machine without API keys or internet.
- [ ] Docker build and fixture execution are documented and reproducible.
- [ ] The README has exact setup, run, test, and local UI steps.
- [ ] Online-only requirements are explicit and failure messages are actionable.

## 6. Proposed v0 repository layout

```text
mlquest/
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── Makefile
├── README.md
├── PROJECT_CONTEXT.md
├── V0_IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── mlquest/
│       ├── cli/
│       ├── config/
│       ├── domain/
│       ├── persistence/
│       ├── artifacts/
│       ├── sessions/
│       ├── phases/
│       ├── budgets/
│       ├── datasets/
│       ├── validation/
│       ├── baselines/
│       ├── experiments/
│       ├── runner/
│       ├── llm/
│       ├── research/
│       ├── reporting/
│       ├── packaging/
│       └── web/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── e2e/
├── docker/
│   └── experiment/
├── templates/
│   ├── task.toml
│   └── reports/
└── docs/
    ├── architecture.md
    ├── security.md
    └── reproducibility.md
```

The exact module names may change during implementation, but the separation between domain contracts, orchestration, execution, evidence, and delivery should remain.

## 7. Definition of done for v0

v0 is complete when a user can:

- [ ] Install the project from a clean Linux machine with the pinned runtime and lockfile.
- [ ] Run the CLI against a local tabular classification or regression task.
- [ ] Provide a problem statement, local train/test files, metric, and budget.
- [ ] Preserve the original data and generate a machine-readable profile.
- [ ] Review or approve the proposed validation strategy.
- [ ] Obtain a reproducible trivial/simple/strong baseline.
- [ ] Run online research through the configured research provider.
- [ ] See research-derived hypotheses linked to planned experiments.
- [ ] Run bounded experiments in Docker with recorded resource use.
- [ ] Resume after a pause or process restart.
- [ ] See which changes improved, failed, or remain unconfirmed.
- [ ] Confirm a candidate using the configured evidence policy.
- [ ] Generate a final solution script/notebook and submission artifact.
- [ ] Manually record an external score without automatic submission.
- [ ] Re-run final verification from the recorded configuration.
- [ ] Generate a complete Markdown/HTML solution guide.
- [ ] View and control the session through a localhost interface.
- [ ] Run the entire offline fixture workflow with `make test`.

## 8. Implementation order recommendation

The work should be executed in these working steps, with a commit after each step that passes its checks:

- [x] Repository/bootstrap documentation.
- [ ] Pinned runtime and project bootstrap — implementation complete; Docker verification blocked by local permissions.
- [x] Domain models and validation contracts.
- [ ] SQLite ledger and artifact store.
- [ ] CLI shell and session state machine.
- [ ] Dataset ingestion and profiling.
- [ ] Metrics, splits, and baselines.
- [ ] Budget manager and resource detector.
- [ ] Docker runner.
- [ ] OpenRouter provider adapter.
- [ ] fastCRW research adapter.
- [ ] Planner and experiment loop.
- [ ] Confirmation, packaging, and verification.
- [ ] Evidence-driven reports.
- [ ] Localhost UI.
- [ ] End-to-end fixture, docs, and final hardening.

No step should be considered complete until its focused tests pass and the repository remains buildable from a clean environment.

## 9. Known assumptions and gaps

- “No maximum dataset size” means no artificial product cap; the system still has to respect local disk, memory, CPU, and model constraints.
- The first implementation may need to use bounded sampling for profiling and must disclose that sampling in reports.
- OpenRouter free models may be rate-limited, removed, or unavailable. Provider/model selection and failure handling must therefore be configurable.
- fastCRW is promising for local or self-hosted research, but its deployment mode, API surface, and AGPL-3.0 licensing implications must be verified before it is bundled into the project distribution.
- Internal validation remains the main optimisation signal unless the user explicitly supplies a labelled evaluation dataset.
- A user-reported leaderboard score is useful evidence for competition work but is not independently reproducible by the local agent unless the relevant labels or evaluation service are available.
- The v0 UI is a control and observability surface, not a replacement for the CLI or the persisted workflow engine.
