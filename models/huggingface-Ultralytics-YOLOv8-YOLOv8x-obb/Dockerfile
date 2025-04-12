FROM python3.12_ai_service_base:latest

WORKDIR /app

COPY . .

RUN pip install ultralytics fastapi uvicorn

RUN curl -L -o yolov8x-obb.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-obb.pt

EXPOSE 8000

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]