FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install future_sales_prediction_2024 

EXPOSE 8000

CMD ["sh", "-c", "echo 'FastAPI is running at http://localhost:8000/docs'; uvicorn app:app --host 0.0.0.0 --port 8000"]