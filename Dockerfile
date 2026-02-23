FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/backend/video_jobs

WORKDIR /app/backend

CMD ["sh", "-c", "uvicorn main:ismayil_server --host 0.0.0.0 --port ${PORT:-8000}"]
