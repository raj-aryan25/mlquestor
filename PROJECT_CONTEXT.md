# Autonomous ML Competition Research Agent — Project Context

> Living context document for the project idea. This document records the intended product, scope, decisions, and constraints before implementation begins. It is not an implementation plan or code specification yet.

## 1. Project vision

Build an autonomous, search-driven machine-learning research agent for Kaggle-style competitions and other well-defined ML challenges.

The agent should accept a problem description, dataset, evaluation metric, and computational budget. It should then understand the task, analyse the data, establish a trustworthy baseline, research possible approaches, generate and execute targeted experiments, learn from measured outcomes, and adapt its search strategy within the available budget.

The central requirement is **evidence-backed improvement**. The agent must not merely suggest a model or claim that one approach is better. Every reported improvement must be supported by recorded, reproducible experimental evidence.

The final output should be a verified competition-ready solution, together with the code, configuration, experiment history, and a complete solution guide.

## 2. Initial product focus

### Primary use case

The initial system is primarily a **competition-solving and ML research-and-development system** for:

- Kaggle-style competitions;
- well-defined supervised ML challenges;
- problems with a known dataset and evaluation metric;
- situations where a solution must be produced within a limited local compute window.

The design may later expand to broader ML research and production use cases, but the first version should optimise for competition workflows and reproducibility.

### Initial data domain

The first supported domain is **tabular data**.

The initial scope should prioritise tabular classification and regression. Other domains such as time series, NLP, computer vision, audio, and multimodal learning are future extensions rather than initial requirements.

### Initial execution environment

- Experiments run on the user's local machine.
- CPU execution is required.
- GPU support may be added when a compatible GPU is available, but GPU availability must not be assumed.
- The system should detect available resources and adapt execution accordingly.

## 3. User inputs

The system should eventually accept the following inputs:

1. **Problem statement**
   - Natural-language description of the challenge;
   - task type and prediction objective, when known;
   - domain context and known restrictions.

2. **Dataset**
   - Training data;
   - test data, when applicable;
   - optional sample submission;
   - optional data dictionary or metadata.

3. **Evaluation definition**
   - Primary metric;
   - metric direction (maximise or minimise);
   - optional secondary metrics;
   - validation or submission requirements.

4. **Computational budget**
   - Total competition-session budget;
   - model training and inference budget;
   - optional experiment-count, memory, storage, or monetary limits.

5. **Operating mode**
   - Fully autonomous execution; or
   - approval-gated execution at configured workflow phases.

## 4. Budget interpretation for the initial version

The initial default budget model has two levels:

### Total session budget

The complete competition-solving session should fit within approximately **two hours**. This includes:

- problem and dataset analysis;
- web research;
- experiment planning;
- model training and inference;
- result analysis;
- final verification;
- report and submission generation.

### Model execution budget

The training and inference code for the solution should fit within approximately **90 minutes** of runtime.

These are separate but related constraints. The system must track both and must not consume the entire session on early experiments. It should reserve time for final confirmation, clean reruns, submission generation, and report creation.

The values should eventually be configurable rather than hard-coded. The budget manager should expose remaining wall-clock time and estimated cost before starting expensive work.

## 5. Autonomy and workflow phases

The agent should support both of these modes:

### Fully autonomous mode

The system proceeds through the workflow without requiring approval, while respecting the configured budget, safety limits, and execution policy.

### Approval-gated mode

The system pauses before selected phases and asks the user to approve the next step. The workflow should be divided into phases so the user can approve a meaningful unit of work rather than every individual command.

The initial conceptual phases are:

1. **Intake phase**
   - Parse the challenge description;
   - identify missing information and assumptions;
   - create a structured task specification.

2. **Dataset phase**
   - Inspect schema, distributions, missingness, duplicates, and data quality;
   - identify possible leakage and split concerns;
   - propose a validation strategy;
   - produce a data-analysis summary.

3. **Baseline phase**
   - Establish trivial, simple, and credible tabular baselines;
   - validate the metric and evaluation pipeline;
   - record baseline runtime and quality.

4. **Research phase**
   - Search approved web sources for papers, documentation, implementations, and relevant techniques;
   - extract actionable ideas;
   - link each idea to a testable hypothesis and source reference.

5. **Experiment phase**
   - Prioritise and execute targeted experiments;
   - track resources, configurations, code versions, and outcomes;
   - update the search strategy using measured results.

6. **Confirmation phase**
   - Re-test promising candidates with stronger evidence;
   - use repeated seeds, cross-validation, ablations, or clean reruns where appropriate;
   - check that claimed improvements are stable and meaningful.

7. **Finalisation phase**
   - Freeze the selected solution;
   - generate the final training/inference or submission artifact;
   - include preprocessing and feature transformations in the final solution code.

8. **Verification and reporting phase**
   - Re-run the final solution from the recorded configuration;
   - validate the submission format and output;
   - generate the final report and reproducibility guide.

The exact phase boundaries may evolve, but the system should preserve phase-level resumability and approval decisions.

## 6. Dataset handling policy

The original input data must be preserved and treated as immutable.

The agent may create derived data for experiments, including:

- cleaned data;
- imputed data;
- encoded categorical features;
- engineered features;
- filtered or transformed views;
- fold-specific training data;
- cached intermediate artifacts.

However:

- the original data must never be overwritten;
- every derived dataset should be traceable to its source and transformation;
- the final solution must include the preprocessing and feature-engineering logic required to recreate the derived data;
- transformations must be applied consistently between training and inference/submission;
- test data must not be used improperly for target-aware decisions.

The system should explicitly distinguish between:

- the immutable source dataset;
- experiment-specific derived data;
- the final reproducible preprocessing pipeline;
- cached artifacts that can be regenerated.

## 7. Web research requirements

Web research is an important capability, not an optional convenience.

The agent should be able to research:

- academic papers;
- official library and framework documentation;
- competition discussions and public solution write-ups;
- reference implementations and code repositories;
- domain-specific modelling techniques;
- known approaches to similar datasets or metrics.

Research should be evidence-oriented. For each useful finding, the system should preserve:

- source title and URL;
- source type;
- relevant passage or extracted idea;
- date accessed, where practical;
- the assumptions and limitations of the approach;
- the experiment or hypothesis derived from it.

Web research should not be treated as proof that a technique will work on the current dataset. It should generate candidate hypotheses that must be tested locally.

## 8. Baseline and experiment principles

The baseline is a required stage, not an optional convenience.

The agent should first establish whether:

- the task is learnable;
- the metric is implemented correctly;
- the validation strategy is plausible;
- the data contains leakage or severe quality issues;
- a simple model is already competitive;
- the expected runtime is feasible.

Experiments should be targeted and hypothesis-driven where possible. Each experiment should record:

- a unique experiment ID;
- its parent or comparison baseline;
- the hypothesis being tested;
- dataset and code versions;
- preprocessing and feature set;
- model family and configuration;
- random seed(s);
- validation strategy;
- primary and secondary metrics;
- runtime and resource use;
- failure information, if applicable;
- the conclusion and next suggested action.

The system should avoid treating an improvement from a different split, different data, or different metric implementation as a valid comparison.

## 9. Reproducibility expectations

Reproducibility is a core requirement because competition solutions are expected to be rerunnable and auditable.

The final solution should be reproducible from configuration. Depending on the challenge, reproducibility may mean exact or near-exact reproduction of the result rather than bit-for-bit identity, but the system must state which standard is being used.

The final package should aim to include:

- a notebook and/or Python script;
- training and inference/submission entry points;
- preprocessing and feature-engineering code;
- configuration files;
- random seeds;
- dependency and runtime information;
- dataset and artifact references or hashes;
- the evaluation command;
- expected output format;
- recorded metrics and tolerances;
- a reproduction guide.

Every claimed improvement should be linked to the experiment that produced it. The final report should be generated from the experiment records rather than relying only on the agent's conversational memory.

## 10. Final deliverables

The final output may include one or more of the following, depending on the challenge:

- competition submission file;
- reproducible Python script;
- reproducible notebook;
- trained model artifact;
- inference script;
- experiment ledger;
- dataset analysis report;
- research/source summary;
- final model-selection summary;
- verification result;
- complete solution guide;
- limitations and future-work section.

The solution guide should explain:

1. What the problem was;
2. What was discovered about the data;
3. How the validation strategy was selected;
4. What the baseline achieved;
5. Which experiments were run;
6. Which changes improved or harmed performance;
7. Why the final approach was selected;
8. How to reproduce the result;
9. What constraints and limitations remain.

## 11. Quality, safety, and trust requirements

The system should actively detect and report:

- target, temporal, group, and preprocessing leakage;
- duplicate records across splits;
- validation overfitting;
- class imbalance and unsuitable metrics;
- inconsistent train/inference preprocessing;
- unstable seed-dependent results;
- resource exhaustion and timeouts;
- failed or incomplete experiments;
- unsupported claims in the final report.

The agent should be able to say that there is insufficient evidence for an improvement. A result should not be accepted solely because it has a higher single-run score.

Generated experiment code should eventually run in a controlled environment with resource limits, timeouts, and restrictions on filesystem and network access.

## 12. Initial non-goals

The first version does not need to solve all of the following:

- every ML data domain;
- distributed training;
- guaranteed GPU execution;
- unrestricted autonomous software engineering;
- production model deployment;
- fully autonomous external-data acquisition;
- bit-for-bit reproducibility on every hardware platform;
- competition-specific optimisation for every possible platform;
- human-free decision making for high-risk or expensive actions.

These may be considered after the tabular CPU workflow is reliable.

## 13. Long-term expansion areas

Potential future extensions include:

- time series, NLP, computer vision, audio, and multimodal adapters;
- GPU and distributed execution;
- cloud or cluster backends;
- reusable experiment and dataset registries;
- richer AutoML and multi-fidelity search;
- ensemble and stacking discovery;
- collaborative research workspaces;
- competition-specific plugins;
- production packaging and deployment;
- broader scientific research workflows.

## 14. Current guiding principle

The product should be designed as a **budget-aware, evidence-driven experimentation system with an intelligent research and planning layer**.

The agent may propose ideas, write solution code, research the web, and adapt its search. The execution and evidence layers must remain authoritative so that the final result is not just plausible, but measurable, reproducible, and verifiable.

## 15. Current status

- Project idea and initial scope captured.
- Initial domain: tabular ML.
- Initial execution: local CPU, with optional GPU support later.
- Initial target: Kaggle-style and well-defined ML challenges.
- Research: web research is required.
- Default session budget: approximately two hours.
- Default solution training/inference budget: approximately 90 minutes.
- Autonomy: fully autonomous and approval-gated modes are both required.
- Data policy: preserve original data; derived experiment data is allowed; final code must contain required preprocessing.
- Final output: notebook and/or Python solution plus reproducibility materials and a complete guide.
- Implementation has not started.

## 16. Confirmed decisions for v0

The following decisions were confirmed after the initial architecture and technology-stack discussion:

- The initial LLM provider path should support **OpenRouter**, with free models used where available. The provider integration must remain model- and provider-configurable because free-model availability, quotas, and rate limits can change.
- The application should run in **online mode only** for v0 because LLM reasoning and web research depend on online providers. Local ML experiment execution remains local and should not require network access inside the experiment sandbox.
- Docker is available on the current development machine and should be used for controlled experiment execution.
- The primary interaction model should be an **agent-style CLI** that works against local files and datasets. A localhost visual interface should also be provided, initially as a view/control surface over the same session and ledger rather than as a separate execution engine.
- There should be no artificial product-level maximum dataset size. Practical limits are determined by the user's disk, memory, CPU, file formats, and model requirements. Large-data profiling and ingestion should be resource-aware rather than loading everything into an LLM prompt or assuming the entire dataset fits in memory.
- The web research layer should initially evaluate **fastCRW/crw** as a local or reachable search/scrape backend. It must be accessed through a replaceable research-provider adapter, and its licensing/deployment implications must be reviewed before bundling or redistributing it.
- The v0 system should not automatically submit to Kaggle or other competition platforms.
- Evaluation should support multiple evidence sources:
  - internal validation scores generated by the agent;
  - scores on a user-provided labelled holdout/test dataset, when available;
  - externally reported competition or test scores entered by the user after manual evaluation.
- Internal validation remains the default basis for experiment selection. User-reported external scores are recorded as external observations and must not be confused with locally reproduced evidence.
