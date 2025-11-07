#!/bin/bash

# Start the curly-giggleAI application
cd /Users/arzt/giggiliai/curly-giggleAI

# Run the Python app
/Users/arzt/giggiliai/curly-giggleAI/venv/bin/python app.py &

# Wait a moment for the server to start
sleep 3

# Open the browser
open https://localhost:8031
