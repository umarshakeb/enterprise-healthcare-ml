FROM python:3.11-slim

# Install system utilities needed for DVC execution
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all project and ML pipelines dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn "dvc[s3]" awscli

# Copy your full codebase architecture into the container image
COPY . .

# Convert execution permissions for our script runner
RUN chmod +x run.sh

# Force Gradio to listen to Render's dynamic global port allocation variable
ENV PORT=10000
EXPOSE 10000

# Execute the dual server launch script
CMD ["./run.sh"]
