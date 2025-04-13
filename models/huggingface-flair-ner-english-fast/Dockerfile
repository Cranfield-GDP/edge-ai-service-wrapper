FROM python3.12_ai_service_base:latest

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install additional dependencies
RUN pip install --no-cache-dir flair

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]