FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
COPY preprocessing.py .
COPY features.py .
COPY model.py .
CMD ["python", "-u", "handler.py"]
