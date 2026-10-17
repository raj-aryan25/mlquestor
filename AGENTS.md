# Agent Instructions

## Python environment

- Use the project-local UV-managed virtual environment at `.venv` for all Python commands.
- Use the committed `uv.lock` and run `uv sync --frozen --extra dev` to create or synchronize the development environment.
- Do not use the system Python, Conda, or create a different environment unless explicitly requested.
- Verify that `.venv/bin/python` is available and reports the pinned Python 3.12 runtime before running Python tools, scripts, or tests.
- Prefer `uv run --locked ...` or the equivalent `.venv/bin/...` command so dependency resolution remains reproducible.

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
- Run the relevant project tests after each development step to verify the changes.
- Running tests does not require explicit user approval.

## Git workflow

- Ask the user for approval before committing code.
- Ask the user for approval before pushing code.
- Before requesting commit approval, inspect the working tree and review the intended diff.
- Run the relevant test command again immediately before an approved commit.
- Never commit or push without explicit user approval.
- Do not include secrets, credentials, local datasets, generated caches, or unrelated changes in commits.
