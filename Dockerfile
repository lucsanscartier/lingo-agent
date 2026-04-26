# ── LINGO Dockerfile ───────────────────────────────────────────────────────
# Targets Hugging Face Spaces (CPU, Python 3.11-slim base).
# Build: docker build -t lingo-backend .
# Run:   docker run --env-file .env -p 7860:7860 lingo-backend

FROM python:3.11-slim

LABEL maintainer="LINGO"
LABEL description="LINGO AI Phone Agent Backend"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HEALTH_ENABLED=true \
    MEMORY_BACKEND=sqlite \
    MEMORY_DB=/data/lingo_memory.sqlite3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash lingo \
    && mkdir -p /app /data \
    && chown -R lingo:lingo /app /data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY agent.py config.py health_server.py memory.py prompts.py ./

USER lingo

EXPOSE 7860
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["python", "agent.py"]
