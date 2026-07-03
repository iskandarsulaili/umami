# Umami ML Service Dockerfile
# GPU-capable with NVIDIA CUDA 13.x support
# Falls back to CPU if no GPU available

FROM nvidia/cuda:13.3-runtime-ubuntu24.04 AS base

LABEL description="Umami ML Service - GPU-accelerated ML inference"
LABEL maintainer="umami-software"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY ml/requirements.txt /app/ml/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/ml/requirements.txt

# Copy ML service code
COPY ml/ /app/ml/
COPY ml/pyproject.toml /app/

# Install the ML package
RUN pip install -e /app

WORKDIR /app

# Expose ML API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

# Default: run ML API server
CMD ["python3", "-m", "ml.api.main"]
