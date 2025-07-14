# MedVision Classification Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/outputs/checkpoints /app/outputs/logs

# Expose port for any web services (if needed)
EXPOSE 8888

# Default command
CMD ["python", "-m", "medvision_cls.cli.main", "--help"]
