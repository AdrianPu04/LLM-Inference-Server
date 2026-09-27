FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SERVER=continuous_batch_server \
    PORT=8000

WORKDIR /app

RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server/ server/
COPY scripts/ scripts/

# GKE mounts the host NVIDIA driver here; CUDA base images set this, python:slim does not.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    PATH=/usr/local/nvidia/bin:$PATH

EXPOSE 8000
VOLUME ["/models"]

CMD exec uvicorn server.${SERVER}:app --host 0.0.0.0 --port ${PORT}
