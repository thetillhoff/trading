FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Note: Scraper data will be mounted at runtime via volume mount
# Default command (can be overridden)
CMD ["python", "visualize_djia.py", "--help"]
