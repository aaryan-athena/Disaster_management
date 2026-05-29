FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads

EXPOSE 7860
CMD ["gunicorn","--bind","0.0.0.0:7860","--workers","1","--threads","4","--timeout","300","--worker-class","sync","--log-level","info","app:app"]
