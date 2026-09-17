FROM python:3.11-slim

# ffmpeg for pulling audio out of the RTSP stream, tzdata so TZ env var works
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bark_detector.py .

# Model files get downloaded on first run into /app/models (mount as a
# volume so you don't re-download on every container restart)
VOLUME ["/app/models"]

CMD ["python", "bark_detector.py"]
