FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY glb_service.py .
ENV PORT=8000
CMD uvicorn glb_service:app --host 0.0.0.0 --port ${PORT}
