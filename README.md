# MLQuest

MLQuest is a budget-aware, evidence-driven autonomous ML competition research agent.

The initial release targets local Kaggle-style tabular classification and regression tasks. It analyses local data, establishes a reproducible baseline, performs online research, proposes targeted experiments, executes them in Docker, records evidence, verifies a final solution, and generates a complete solution guide.

## Current status

The repository is in active v0 implementation. The product contract, project bootstrap, domain contracts, persistence, dataset profiling, CLI session lifecycle, authoritative evaluation layer, and budget/resource management are implemented. Autonomous experiments, verification, reporting, and the localhost UI remain under development.

Authoritative planning documents:

- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — product vision, scope, and confirmed decisions.
- [`V0_IMPLEMENTATION_PLAN.md`](V0_IMPLEMENTATION_PLAN.md) — implementation sequence and acceptance criteria.
- [`AGENTS.md`](AGENTS.md) — repository instructions for future development.

## v0 scope

- Tabular supervised classification and regression.
- CSV and Parquet input.
- Local CPU execution, with GPU detection reserved for later work.
- Agent-style CLI for local files and sessions.
- Localhost visual interface over the same session state.
- OpenRouter as the first online LLM provider.
- fastCRW/crw behind a replaceable web-research provider interface.
- Docker-isolated experiment execution.
- Approximately 120 minutes for a complete session by default.
- Approximately 90 minutes for cumulative model training and inference by default.
- Reproducible baselines, experiments, verification, and reports.
- Manual external-score recording only; no automatic competition submission.

## Design principles

1. The LLM proposes; typed application services validate and execute.
2. The original source data is immutable.
3. Every derived dataset, experiment, score, and claim is traceable.
4. Budgets are enforced before work starts and protected finalisation time is reserved.
5. Web research produces cited hypotheses, not proof of local improvement.
6. Reports are generated from the persisted evidence ledger.
7. External scores are distinguished from locally reproducible validation evidence.

## Planned first user flow

```text
Create local project
  ↓
Describe task, data, metric, and budget
  ↓
Start an autonomous or approval-gated session
  ↓
Inspect data findings and validation proposal
  ↓
Run reproducible baselines
  ↓
Research relevant methods online
  ↓
Run bounded Docker experiments
  ↓
Confirm the strongest candidate
  ↓
Verify and package the final solution
  ↓
Read the generated report and submit manually if desired
```

## Planned local project layout

An MLQuest task project is expected to look like this:

```text
competition-project/
├── mlquest.toml             # task, metric, budget, and policy
├── data/
│   ├── train.csv            # user-provided source data
│   ├── test.csv             # optional competition test data
│   └── sample_submission.csv
├── .mlquest/
│   └── sessions/
│       └── <session-id>/
│           ├── ledger.sqlite
│           ├── manifest.json
│           ├── sources/
│           ├── derived/
│           ├── artifacts/
│           ├── reports/
│           └── final/
└── solution/                # optional user-managed exports
```

The `data/` files are never modified by MLQuest. Session-specific copies or references are registered as immutable source artifacts. Generated data, models, logs, reports, and final bundles are kept under `.mlquest/sessions/<session-id>/`.

## Planned task configuration

The v0 task configuration will be TOML-based and validated before a session starts. A representative configuration is:

```toml
[task]
name = "example-tabular-task"
problem_statement = "Predict the target for each row in the competition test set."
task_type = "classification"
target_column = "target"

[data]
train = "data/train.csv"
test = "data/test.csv"
sample_submission = "data/sample_submission.csv"

[evaluation]
metric = "roc_auc"
direction = "maximize"
validation_strategy = "stratified_kfold"
n_splits = 5
seed = 42

[budget]
session_minutes = 120
execution_minutes = 90

[policy]
mode = "approval"
automatic_submission = false
```

The actual schema may evolve during Step 2, but unsupported or ambiguous fields must be rejected or surfaced for approval instead of silently guessed.

## Planned CLI vocabulary

The CLI is the primary interface. These commands define the intended v0 workflow:

```text
mlquest init PATH
mlquest start --project PATH --task mlquest.toml
mlquest status --project PATH --session SESSION_ID
mlquest approve --project PATH --session SESSION_ID
mlquest reject --project PATH --session SESSION_ID
mlquest pause --project PATH --session SESSION_ID
mlquest resume --project PATH --session SESSION_ID
mlquest experiments list --project PATH --session SESSION_ID
mlquest experiments show EXPERIMENT_ID --project PATH --session SESSION_ID
mlquest research list --project PATH --session SESSION_ID
mlquest score add --project PATH --session SESSION_ID
mlquest verify --project PATH --session SESSION_ID
mlquest report --project PATH --session SESSION_ID
mlquest package --project PATH --session SESSION_ID
mlquest ui --project PATH --session SESSION_ID
```

The exact option names are implementation details, but the command responsibilities should remain stable. CLI and localhost UI operations must call the same service layer and cannot bypass approvals, budgets, validation, or artifact-scope checks.

## Evaluation and score provenance

MLQuest distinguishes three score sources:

### Internal validation

The agent computes this score using the configured split, metric, and validation pipeline. It is the default score used for experiment comparison and candidate selection.

### User-provided labelled holdout/test data

If the user supplies a labelled evaluation dataset and explicitly identifies it, MLQuest can evaluate the final pipeline locally. The dataset and evaluation configuration are recorded as separate evidence inputs.

### User-reported external score

After manually evaluating or submitting a generated artifact elsewhere, the user may record the score using `score add`. MLQuest stores the value, metric, source, timestamp, and notes. It does not upload the submission and does not treat the score as a locally reproduced result.

## Initial metric registry

The first metric registry is intentionally small and explicit:

### Classification

- `accuracy`
- `balanced_accuracy`
- `f1`
- `precision`
- `recall`
- `roc_auc`
- `log_loss`

### Regression

- `mae`
- `mse`
- `rmse`
- `r2`

Each metric has a fixed implementation, supported prediction type, direction, and validation rules. Custom metrics are not implicitly executed in v0; they require an explicit validated extension path.

## Online provider setup

MLQuest v0 requires online access for LLM reasoning and web research.

Planned provider configuration:

```text
OPENROUTER_API_KEY=replace-with-local-secret
MLQUEST_LLM_MODEL=replace-with-configured-openrouter-model
```

Secrets must be supplied through the environment or an ignored local `.env` file. They must never be committed, copied into task files, sent in prompts, or stored in reports.

Free OpenRouter models are supported where available, but availability, quotas, and rate limits can change. The model must therefore remain configurable and provider failures must be reported clearly.

The initial research adapter will evaluate fastCRW/crw. It will be isolated from experiment execution, and retrieved web pages will be treated as untrusted input. Research code will not be downloaded and executed automatically.

## Planned setup, run, test, and local deployment commands

These commands are the target v0 interface. They will become executable as the corresponding implementation steps are completed.

### Setup

```bash
conda activate mlquestor
uv sync --frozen --extra dev
cp .env.example .env
```

Docker must be installed and available to the current user. The OpenRouter key and research-provider configuration are required for online phases.

### Run a session

```bash
conda activate mlquestor
mlquest init ./competition-project
mlquest start --project ./competition-project --task mlquest.toml
```

Use approval-gated mode for the first manual run. Autonomous execution should be enabled only after the task configuration and validation proposal are understood.

### Run tests

```bash
conda activate mlquestor
make test
```

The test command must work offline. OpenRouter and fastCRW calls will be mocked in tests. No credentials should be required for the test suite.

### Start the localhost interface

```bash
conda activate mlquestor
mlquest ui --project ./competition-project --session SESSION_ID
```

The v0 interface will bind to localhost by default and will expose the same persisted state as the CLI.

### Build the application image

```bash
docker build -t mlquest:dev .
```

Build the isolated experiment image from the same pinned project dependencies:

```bash
docker build -f docker/experiment/Dockerfile -t mlquest-experiment:dev .
```

The application image and experiment image must be built from pinned project dependencies. Experiment containers must not receive general outbound network access.

### Manual competition submission

MLQuest generates the final script, notebook, prediction file, and solution guide. The user submits artifacts manually to Kaggle or another platform and can then record the external score with `mlquest score add`.

## v0 non-goals

- Automatic competition submission.
- Distributed or cloud experiment execution.
- Required GPU support.
- Deep-learning model families.
- Time-series, NLP, computer-vision, or multimodal adapters.
- Unrestricted arbitrary shell execution on the host.
- Treating web research claims as experimental evidence.

## Development status

Steps 0 through 8 are implemented. The project now has a verified Python 3.12.x bootstrap, domain contracts, a SQLite ledger, an immutable content-addressed artifact store, dataset ingestion/profiling services, a durable CLI session lifecycle, reproducible metrics/splits/prediction validation/baselines, resource detection, durable budget admission/accounting, and an application-integrated isolated Docker experiment runner. Later verification, reporting, packaging, and UI commands are exposed but their underlying services are still pending.
