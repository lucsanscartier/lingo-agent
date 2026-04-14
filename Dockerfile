# ── LINGO Dockerfile ──────────────────────────────────────────────────────────
# Targets Hugging Face Spaces (CPU, Python 3.11-slim base).
# Build: docker build -t lingo-backend .
# Run:   docker run --env-file .env lingo-backend

FROM python:3.11-slim

# Metadata
LABEL maintainer="LINGO"
LABEL description="LINGO AI Phone Agent Backend"

# System dependencies required for soundfile (libsndfile) and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash lingo
WORKDIR /app

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY agent.py memory.py prompts.py ./

# HF Spaces expects the app to listen on port 7860 if it runs a server.
# LINGO is a worker process (no HTTP server), so no EXPOSE is needed.
# If you add a health-check HTTP endpoint later, uncomment:
# EXPOSE 7860

# Switch to non-root user
USER lingo

# The memory file will be written to /app by default; make sure the volume
# or HF Space persistent storage is mounted here if you want persistence.
VOLUME ["/app"]

CMD ["python", "agent.py"]
