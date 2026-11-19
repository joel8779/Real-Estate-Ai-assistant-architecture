# ==============================================================================
# DarGlobal & Wasalt AI Assistant - Backend Dockerfile (Python 3.11)
# Implements non-root user execution and multi-stage security best practices.
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch CPU wheel with retries and timeout for resilient caching
RUN pip install --default-timeout=120 --retries 5 --no-cache-dir --user torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --default-timeout=120 --retries 5 --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


# Production Stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy installed packages from builder to non-root location
COPY --chown=appuser:appgroup --from=builder /root/.local /home/appuser/.local

# Copy application source and persistent directory scaffold
COPY --chown=appuser:appgroup app ./backend/app

# Ensure proper permissions for application and data directories
RUN mkdir -p ./data/raw/darglobal ./data/raw/wasalt ./data/processed ./data/chroma && \
    chown -R appuser:appgroup /app

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Health check using liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
