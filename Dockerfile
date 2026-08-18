FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mtp --no-editable

FROM python:3.13-slim
RUN useradd --system --uid 10001 --create-home swp
RUN install -d -o swp -g swp /var/lib/swp
ENV HF_HOME=/var/lib/swp/_hf_runtime \
    HF_XET_CACHE=/var/lib/swp/_hf_runtime/xet \
    XDG_CACHE_HOME=/var/lib/swp/_xdg_cache
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
USER 10001
ENTRYPOINT ["/app/.venv/bin/skulk-swp-worker"]
