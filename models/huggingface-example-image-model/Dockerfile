# Do Not Separate Builder and Runtime Stages
FROM python:3.12-slim 

# Install common dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch \
    torchvision \
    opencv-python \
    opencv-python-headless \
    accelerate \
    transformers \
    python-multipart \
    Pillow \
    psutil \
    requests

# Install XAI libraries.
# Currently only XAI for image-classification models in pytorch is supported.
RUN pip install --no-cache-dir grad-cam

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
