FROM python3.12_ai_service_base:latest

WORKDIR /app

COPY . .

RUN pip install transformers

EXPOSE 8000

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]