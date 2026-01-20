# DJIA Scraper

Downloads and caches DJIA (Dow Jones Industrial Average) historical data from Yahoo Finance.

## Overview

This scraper downloads DJIA data starting from 1900-01-01 and caches it locally in CSV format. On
subsequent runs, it loads from the cached file instead of re-downloading, making it fast and
efficient.

## Features

- Downloads DJIA historical data from yfinance
- Caches data locally in `djia_data.csv`
- Automatically loads from cache if available
- Displays data summary (shape, date range, sample rows)

## Requirements

- Docker (recommended)
- Or Python 3.11+ with dependencies from `requirements.txt`

## Usage

### Using Docker (Recommended)

```bash
# Build the Docker image
make build

# Run the scraper
make run

# Or build and run in one command
make build && make run
```

### Using Docker directly

```bash
# Build
cd scrapers/djia
docker build -t djia-scraper .

# Run
docker run --rm -v $(pwd):/app djia-scraper
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper
python download_djia.py
```

## Output

The scraper generates `djia_data.csv` in the same directory as the script. This file contains:

- Date index
- OHLCV columns: Open, High, Low, Close, Volume

## Data Format

The CSV file uses a multi-level header format:

- First row: Column names (Price, Close, High, Low, Open, Volume)
- Second row: Ticker symbols (^DJI)
- Third row: Date header
- Data rows: Date and corresponding values

## Scheduling

This scraper can be scheduled to run automatically (e.g., daily via cron):

```bash
# Example cron job (runs daily at 2 AM)
0 2 * * * cd /path/to/trading/scrapers/djia && make run
```

## Troubleshooting

- **File not found errors**: Ensure you have write permissions in the script directory
- **Network errors**: Check your internet connection and yfinance API availability
- **Docker errors**: Ensure Docker is running and you have permissions
