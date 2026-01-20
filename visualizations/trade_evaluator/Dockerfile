# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel for better package management
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Default command shows help
CMD ["python", "evaluate_trades.py", "--help"]
