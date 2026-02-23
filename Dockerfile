# Python 3.11 slim — kiçik ölçü, tam pip dəstəyi
FROM python:3.11-slim

# Sistem paketlərini quraşdırırıq (ffmpeg video emalı üçün)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# İş qovluğu
WORKDIR /app

# Əvvəlcə requirements-ı kopyalayırıq (Docker cache üçün)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bütün proyekti kopyalayırıq
COPY . .

# video_jobs qovluğunu yaradırıq (mounting üçün)
RUN mkdir -p /app/backend/video_jobs

# Backend qovluğuna keçirik
WORKDIR /app/backend

# PORT env var-ı Railway tərəfindən avtomatik təyin edilir
EXPOSE 8000

CMD uvicorn main:ismayil_server --host 0.0.0.0 --port ${PORT:-8000}
