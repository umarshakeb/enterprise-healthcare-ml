#!/bin/bash
# Start the backend FastAPI engine in the background
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Start the frontend Gradio user interface on Render's assigned port boundary
python ui/gradio_app.py
