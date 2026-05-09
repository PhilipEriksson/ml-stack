#!/usr/bin/env bash
set -e

# Start FastAPI proxy in background on port 8000
echo "Starting FastAPI proxy on port 8000..."
cd /opt/api && nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &

# Set WebUI to point at our proxy
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://localhost:8000/v1}"

# Start Open WebUI - try the image's start script, fallback to uvicorn directly
echo "Starting Open WebUI..."
export PORT=8080
if [ -f /app/backend/start.sh ]; then
    exec /app/backend/start.sh
elif [ -f /app/backend/open_webui/main.py ]; then
    cd /app/backend && exec uvicorn open_webui.main:app --host 0.0.0.0 --port 8080
else
    echo "ERROR: Cannot find Open WebUI entry point"
    exit 1
fi
