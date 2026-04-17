FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py .
COPY handler.py .
COPY model_artifacts/ ./model_artifacts/

ENV MODEL_PATH=/app/model_artifacts/lda_model.pkl
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]