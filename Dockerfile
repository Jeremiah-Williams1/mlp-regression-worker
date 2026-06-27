FROM python:3.11-slim

WORKDIR /app

# git is needed at build time for the `git+https://...` line in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py worker.py ./
COPY model/ ./model/

ENV MODEL_PATH=/app/model/auto_mpg_mlp
ENV FEATURE_COLUMNS_PATH=/app/model/feature_columns.json
# REDIS_URL and PUSHGATEWAY_URL are set via the k8s Deployment/ConfigMap,
# not baked in here -- same pattern as the existing classification worker.

CMD ["python", "worker.py"]
