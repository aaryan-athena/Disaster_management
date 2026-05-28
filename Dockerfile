FROM python:3.11-slim

# System libraries required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure /app is always on Python's module search path so gunicorn can find
# local packages like `ml/` and `db.py` regardless of how it is invoked.
ENV PYTHONPATH=/app

# Install Python deps first — better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p uploads

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--worker-class", "sync", \
     "--log-level", "info", \
     "app:app"]
