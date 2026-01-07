#!/bin/bash
set -e

echo "Starting Ollama setup script..."

# Install curl if not present (often needed for healthchecks in minimal images)
if ! command -v curl &> /dev/null; then
    echo "Installing curl..."
    apt-get update && apt-get install -y curl --no-install-recommends
fi

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags; do
    sleep 5
done

echo "Pulling models..."
# Pull the model requested 
ollama pull qwen2.5:3b || echo "Failed to pull qwen2.5:3b"
ollama pull nomic-embed-text-v2-moe || echo "Failed to pull nomic-embed-text-v2-moe"
ollama pull llama3.2-vision || echo "Failed to pull llama3.2-vision"

echo "Models pulled. Keeping Ollama running..."
wait $OLLAMA_PID
