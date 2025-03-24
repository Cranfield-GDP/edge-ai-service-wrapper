# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch \
    torchvision \
    transformers \
    python-multipart \
    Pillow \
    psutil \
    requests

# Stage 2: Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000"]