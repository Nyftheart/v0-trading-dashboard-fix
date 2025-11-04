# Multi-stage build optimized for Raspberry Pi 3B (ARM architecture)
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set timezone
ARG TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies (minimal for RPi 3B)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY backend/ ./backend/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Expose web dashboard port
EXPOSE 8080

# Health check
# Le healthcheck sera géré par docker-compose si nécessaire

# Default command (can be overridden in docker-compose)
CMD ["python", "scripts/advanced-trading-bot.py"]
