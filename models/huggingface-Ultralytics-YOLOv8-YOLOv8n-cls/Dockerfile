FROM python3.12_ai_service_base:latest

WORKDIR /app

COPY . .

RUN pip install ultralytics fastapi uvicorn

EXPOSE 8000

# Download the YOLOv8n-cls model
RUN curl -L -o YOLOv8n-cls.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]