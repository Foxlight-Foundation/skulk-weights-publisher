FROM python:3.13.17-slim-trixie AS base
# Bootstrap pip using a verified artifact before performing HTTPS installs.
ADD --checksum=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70 https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl /tmp/pip-26.2.2-py3-none-any.whl
RUN python -m pip install --no-index --no-deps /tmp/pip-26.2.2-py3-none-any.whl \
    && rm /tmp/pip-26.2.2-py3-none-any.whl

FROM base AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mtp --no-editable

FROM base
RUN useradd --system --uid 10001 --create-home swp
RUN install -d -o swp -g swp /var/lib/swp
ENV HF_HOME=/var/lib/swp/_hf_runtime \
    HF_XET_CACHE=/var/lib/swp/_hf_runtime/xet \
    XDG_CACHE_HOME=/var/lib/swp/_xdg_cache
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# Validate the bundle actually used by outbound worker requests.
RUN /app/.venv/bin/python -c 'import certifi, ssl; bad={"emSign ECC Root CA - G3", "AAA Certificate Services"}; roots=ssl.create_default_context(cafile=certifi.where()).get_ca_certs(); assert not any(value in bad for root in roots for rdn in root["subject"] for key,value in rdn if key=="commonName")'
USER 10001
ENTRYPOINT ["/app/.venv/bin/skulk-swp-worker"]
