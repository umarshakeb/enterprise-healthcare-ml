#!/bin/bash
set -e
 
# Pull the real .joblib model weights from a plain S3 prefix before booting the app
echo "📦 Downloading model artifacts from S3..."
aws s3 sync s3://amzn-s3-health-care/prod-models ./models
 
# Swapped host flag from 127.0.0.1 to 0.0.0.0 so the frontend can cross the container gateway
echo "🔌 Booting up Backend FastAPI Engine on all network interfaces..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
 
# Start the frontend Gradio user interface on Render's assigned port boundary
echo "🌐 Booting up Frontend Gradio interface..."
python ui/gradio_app.py