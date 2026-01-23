FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if they exist, otherwise install common dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install common dependencies for trading analysis
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    yfinance \
    scipy

# Copy the entire project structure
COPY . .

# Note: The project structure is mounted as volumes at runtime via docker-compose.yml
# This allows live code changes without rebuilding

CMD ["python", "-m", "cli.download", "--help"]
