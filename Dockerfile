FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for chromadb)
# Install system dependencies + Ollama
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull models (so they exist at runtime)
RUN ollama pull qwen2.5:0.5b
RUN ollama pull nomic-embed-text

# Copy your app code
COPY app/ ./app
COPY profile.txt .

# Create persistent directory for Chroma
RUN mkdir -p /data/chroma_db

EXPOSE 8000

CMD ["sh", "-c", "ollama serve & uvicorn app.main:app --host 0.0.0.0 --port 8000"]