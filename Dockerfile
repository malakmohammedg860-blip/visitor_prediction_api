FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN git lfs pull

RUN ls -lh tourism_recommendation_model.pkl

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]