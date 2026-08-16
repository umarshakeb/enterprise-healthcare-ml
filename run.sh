#!/bin/bash

echo "📦 Securely downloading machine learning model artifacts via DVC..."
dvc pull -v

# Start the backend FastAPI engine in the background
echo "🔌 Booting up Backend FastAPI Engine..."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Start the frontend Gradio user interface on Render's assigned port boundary
echo "🌐 Booting up Frontend Gradio interface..."
python ui/gradio_app.py
