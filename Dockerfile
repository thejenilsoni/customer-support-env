# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app/env

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies into a prefix we can copy across
RUN pip install --no-cache-dir --prefix=/install \
        "openenv-core[core]>=0.2.2" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.27.0" \
        "pydantic>=2.0.0" \
        "openai>=1.0.0"


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app/env

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Bring in installed packages
COPY --from=builder /install /usr/local

# Bring in environment code
COPY --from=builder /app/env .

# Make sure the env directory is on PYTHONPATH so `from models import ...` works
ENV PYTHONPATH=/app/env

# Hugging Face Spaces: port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
