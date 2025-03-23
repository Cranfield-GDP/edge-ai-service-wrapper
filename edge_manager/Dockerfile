# Use the official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pymongo

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "edge_manager_server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]