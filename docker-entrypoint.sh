#!/bin/bash
# Docker entrypoint script for graceful initialization and startup

set -e

echo "🚀 Starting curly-giggleAI application..."

# Verify critical files exist
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    exit 1
fi

if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "⚠️  Warning: SSL certificates not found. Generating self-signed certificates..."
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" || true
fi

# Verify and set Hugging Face credentials
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_API_TOKEN" ]; then
    echo "⚠️  Warning: HuggingFace token not set. Model downloads may fail for private models."
else
    # Use HF_TOKEN (standard environment variable for Hugging Face)
    if [ -n "$HUGGINGFACE_API_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
        export HF_TOKEN="$HUGGINGFACE_API_TOKEN"
        echo "✅ HuggingFace token configured from HUGGINGFACE_API_TOKEN"
    else
        echo "✅ HuggingFace token configured"
    fi
fi

# Verify Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set. Gemini API calls will fail."
fi

# Pre-warm Whisper model if not cached
if [ ! -d ".cache/faster-whisper" ] || [ -z "$(ls -A .cache/faster-whisper 2>/dev/null)" ]; then
    echo "📥 Pre-downloading Whisper model (this may take a few minutes on first run)..."
    python -c "from faster_whisper import WhisperModel; WhisperModel('tiny')" 2>/dev/null || echo "⚠️  Skipping model pre-download - models will be downloaded on first use"
    echo "✅ Whisper model ready"
else
    echo "✅ Whisper model cache found"
fi

echo "✨ Starting FastAPI server on https://0.0.0.0:8031"
echo "📊 Health check available at: https://0.0.0.0:8031/api/health"
echo "🎤 Speech transcription API: https://0.0.0.0:8031/api/transcribe"
echo ""

# Run the application with proper signal handling
exec python app.py
