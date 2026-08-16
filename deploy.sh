#!/bin/bash
# =====================================================================
# Enterprise GitOps Local Continuous Deployment Script
# =====================================================================
set -e # Immediately exit if any command fails

echo "🛡️ Starting Production-Grade Hardened Local Deployment..."

# 1. Align your host terminal context to Minikube's internal Docker daemon
echo "🔌 Connecting terminal to Minikube Docker Daemon..."
eval $(minikube -p minikube docker-env)

# 2. Recompile your secure, non-root user containers locally
echo "📦 Rebuilding application container images..."
docker build -f Dockerfile.api -t healthcare-api:latest .
docker build -f Dockerfile.gradio -t healthcare-gradio:latest .

# 3. Apply your Zero-Trust Kubernetes manifest definitions
echo "🧱 Applying security manifests and internal cluster firewalls..."
kubectl apply -f network-policy.yaml
kubectl apply -f deployment-api.yaml
kubectl apply -f deployment-gradio.yaml

# 4. Trigger zero-downtime rolling updates
echo "🚀 Triggering zero-downtime rolling application rollouts..."
kubectl rollout restart deployment/healthcare-api-deployment
kubectl rollout restart deployment/healthcare-gradio-deployment

# 5. Monitor verification status
echo "🔍 Verifying cluster rollout state..."
kubectl rollout status deployment/healthcare-api-deployment --timeout=60s
kubectl rollout status deployment/healthcare-gradio-deployment --timeout=60s

echo "✅ Deployment Successful! Your hardened cluster is active and secure."
