FROM python3.12_ai_service_base:latest

WORKDIR /app

COPY . .

RUN pip install ultralytics

EXPOSE 8000

# Download the YOLOv8n model
RUN curl -L -o YOLOv8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]