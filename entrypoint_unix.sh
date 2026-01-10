#!/bin/bash
set -e

echo "Starting Ollama setup script..."

# Install curl
apt-get update && apt-get install -y curl --no-install-recommends

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 5
done

echo "Pulling models..."
ollama pull gemma3:4b || echo "Failed to pull gemma3:4b"
ollama pull nomic-embed-text-v2-moe || echo "Failed to pull nomic-embed-text-v2-moe"
ollama pull llama3.2-vision || echo "Failed to pull llama3.2-vision"
ollama pull qwen3-vl:8b || echo "Failed to pull qwen3-vl:8b"

echo "Models pulled. Keeping Ollama running..."
wait $OLLAMA_PID
