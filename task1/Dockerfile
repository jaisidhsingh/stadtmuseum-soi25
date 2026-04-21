FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/packages -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/packages /app/packages
ENV PYTHONPATH=/app/packages
COPY . .