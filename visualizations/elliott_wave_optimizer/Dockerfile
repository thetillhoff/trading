# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

# Upgrade pip for faster installs and better wheel support
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install dependencies with BuildKit cache mount for faster rebuilds
# This caches pip's download cache between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: Scraper data and djia module will be mounted at runtime via volume mount
# Default command (can be overridden)
CMD ["python", "optimize_filters.py", "--help"]
