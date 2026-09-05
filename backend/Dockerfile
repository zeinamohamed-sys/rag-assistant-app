# DataMind RAG Assistant - Backend Dockerfile
#
# This container runs ONLY the FastAPI backend. Ollama is expected to run
# separately, either on the host machine (recommended for local demos) or
# as its own container/service — see the root README for details.
#
# Build:  docker build -t datamind-backend ./backend
# Run:    docker run -p 8000:8000 --env-file backend/.env \
#           -v $(pwd)/data:/app/data datamind-backend

FROM python:3.11-slim

WORKDIR /app

# System deps needed by some ML wheels (kept minimal on purpose).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The vector store lives outside the image and is mounted at runtime,
# so the container always uses the latest build without a rebuild.
VOLUME ["/app/data"]

ENV VECTOR_STORE_PATH=/app/data/vector_store

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
