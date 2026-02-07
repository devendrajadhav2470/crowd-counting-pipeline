# Multi-stage Dockerfile for crowd counting inference
# Optimized for CPU inference (no CUDA required)

FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml requirements.txt ./
COPY src/ src/

# Install CPU-only PyTorch + project dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[serve]"

# Copy remaining source code
COPY configs/ configs/
COPY scripts/ scripts/
COPY weights/ weights/

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import crowd_counting; print('ok')" || exit 1

# Default command: launch Gradio demo
CMD ["python", "-m", "crowd_counting.serving.gradio_app", "--port", "7860"]

