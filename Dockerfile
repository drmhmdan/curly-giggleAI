# Multi-stage build for optimized image size and performance

# Stage 1: Builder - Download models and prepare dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies needed for compilation and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper models to reduce first-startup latency
# This significantly improves performance on first run
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny')" || true

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (excluding build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_MODEL_DIR=/app/.cache/faster-whisper

# Copy application files
COPY app.py .
COPY sys_instruct .
COPY static/ static/
COPY cert.pem key.pem ./
COPY docker-entrypoint.sh .

# Create cache directory for models
RUN mkdir -p /app/.cache/faster-whisper && \
    chmod +x docker-entrypoint.sh

# Health check - ensures container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('https://localhost:8031/api/health', verify=False)" || exit 1

# Expose port
EXPOSE 8031

# Run entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
