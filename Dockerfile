FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mtp

FROM python:3.13-slim
RUN useradd --system --uid 10001 --create-home swp
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
USER 10001
ENTRYPOINT ["/app/.venv/bin/skulk-swp-worker"]
