#!/bin/bash
# Docker build and test script

echo "Building Docker image..."
docker build -t property-guru-rag .

echo "Testing Docker image locally..."
docker run -p 10000:10000 \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e GRAANA_GEMINI_API_KEY="${GRAANA_GEMINI_API_KEY}" \
  -e SECRET_KEY="test-secret-key" \
  -e PORT=10000 \
  property-guru-rag

echo "To stop the container:"
echo "docker ps"
echo "docker stop <container_id>"
