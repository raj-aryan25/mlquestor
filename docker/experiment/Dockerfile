FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev \
    && groupadd --system --gid 65532 mlquest \
    && useradd --system --uid 65532 --gid 65532 --no-create-home mlquest

USER 65532:65532

ENTRYPOINT ["uv", "run", "--no-dev"]
