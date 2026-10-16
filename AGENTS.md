# Agent Instructions

## Python environment

- Use the Conda environment `mlquestor` for all Python commands.
- Do not use the system Python or create a different environment unless explicitly requested.
- Verify that `mlquestor` is active before running Python tools, scripts, or tests.

## Code quality

- Keep the code clean, focused, and minimally commented.
- Use comments only to describe a specific approach, non-obvious constraint, or important implementation decision.
- Do not add comments that merely restate what the code does.
- Preserve the existing project structure and avoid unrelated changes.

## Front-end development

- Use PNPM for web-based front-end development when a front-end is needed.
- Do not introduce a second JavaScript package manager for the same front-end.
- Use the `impeccable` skill when designing, redesigning, or polishing interfaces.

## Testing workflow

- Build unit tests after each phase of development.
- Keep tests focused on the behavior introduced or changed in that phase.
- Do not claim tests pass unless they were actually run.
- Run the project's tests only in the pre-commit workflow described below.

## Git workflow

- Ask the user for approval before committing code.
- Ask the user for approval before pushing code.
- Before requesting commit approval, inspect the working tree and review the intended diff.
- After approval, run the relevant test command immediately before committing.
- Never commit or push without explicit user approval.
- Do not include secrets, credentials, local datasets, generated caches, or unrelated changes in commits.
