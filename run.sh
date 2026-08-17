#!/bin/bash
set -e

# Force DVC to download your real .joblib weights from S3 before booting the code
echo "📦 Securely downloading machine learning model artifacts via DVC..."
dvc pull -v

# FIXED: Swapped host flag from 127.0.0.1 to 0.0.0.0 so the frontend can cross the container gateway
echo "🔌 Booting up Backend FastAPI Engine on all network interfaces..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start the frontend Gradio user interface on Render's assigned port boundary
echo "🌐 Booting up Frontend Gradio interface..."
python ui/gradio_app.py