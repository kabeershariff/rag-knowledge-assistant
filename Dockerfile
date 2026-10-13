FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANGSMITH_TRACING=false
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
EXPOSE 8000
# HOST and PORT are read here so the documented settings actually reach Uvicorn.
# exec keeps Uvicorn as PID 1, so it still receives SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\""]