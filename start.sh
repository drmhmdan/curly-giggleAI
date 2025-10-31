#!/bin/bash
# Startup script for the Speech Transcription & AI Response application

set -e

echo "========================================"
echo "Speech Transcription & AI Response"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python $(python3 --version) found"

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed."
    echo "Please install pip (Python package manager)."
    exit 1
fi

echo "✓ pip found"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo ""
    echo "Installing dependencies..."
    echo "This may take a few minutes..."
    
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "Warning: GEMINI_API_KEY environment variable is not set."
    echo "The Gemini AI response feature will not work."
    echo ""
    echo "To set your API key, run:"
    echo "  export GEMINI_API_KEY=\"your-api-key-here\""
    echo ""
    echo "Or create a .env file with:"
    echo "  GEMINI_API_KEY=your-api-key-here"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ GEMINI_API_KEY is set"
fi

echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Start the server
python3 app.py
