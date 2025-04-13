# Base image for AI service powered by HuggingFace pre-trained AI models.
# the image has the following packages/libraries installed already:
# - python3.12, pip, git, fastapi, uvicorn, torch, torchvision, opencv-python, transformers, python-multipart, Pillow, requests
FROM python3.12_ai_service_base:latest

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

RUN apt-get update && apt-get install -y espeak-ng

# Install additional dependencies
RUN pip install kokoro>=0.9.2 soundfile misaki

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]