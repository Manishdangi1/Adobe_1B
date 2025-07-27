# Use Python 3.11 slim image as base for smaller size
FROM python:3.11-slim

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models
ENV CUDA_VISIBLE_DEVICES=""

# Set working directory
WORKDIR /app

# Install system dependencies (minimal set for PDF processing and OCR)
RUN apt-get update && apt-get install -y \
    # PDF processing dependencies
    poppler-utils \
    # OCR dependencies (minimal)
    tesseract-ocr \
    tesseract-ocr-eng \
    # Image processing dependencies (minimal)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    # Build dependencies for Python packages
    build-essential \
    gcc \
    g++ \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Copy optimized requirements first for better caching
COPY requirements-optimized.txt requirements.txt

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean pip cache
    rm -rf ~/.cache/pip

# Create necessary directories
RUN mkdir -p /app/models /app/cache /app/data /app/output /app/logs

# Copy application code
COPY . .

# Set permissions
RUN chmod +x generate_solution.py enhanced_solution.py optimize_models.py process_collections.py

# Pre-download and optimize superior models for offline use
RUN python optimize_models.py

# Expose port for FastAPI (if needed)
EXPOSE 8000

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - process all collections for Challenge 1B
CMD ["python", "process_collections.py"] 