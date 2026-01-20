# Trading Analysis Project

A comprehensive Python project for analyzing DJIA (Dow Jones Industrial Average) trading data
using Elliott Wave Theory, signal detection, and trade evaluation.

## Overview

This project provides a complete toolkit for:

- **Data Collection**: Downloading and caching historical DJIA data
- **Elliott Wave Analysis**: Detecting wave patterns in price movements
- **Trading Signals**: Identifying buy/sell opportunities with target prices and stop-loss levels
- **Trade Evaluation**: Backtesting signals to calculate performance metrics
- **Visualization**: Generating charts and graphs for analysis

## Quick Start

See all available commands:

```bash
make help
```

For detailed examples and best practices, see [EXAMPLES.md](EXAMPLES.md).

## Running with Docker

### Using Makefile (Recommended)

```bash
make up        # Build and run using docker-compose
make run       # Build and run using Docker directly
make down      # Stop and remove containers
make clean     # Remove containers, images, and volumes
make rebuild   # Clean and rebuild everything
```

### Using Docker Compose

```bash
docker-compose up --build
```

### Using Docker directly

Build the image:

```bash
docker build -t trading-app .
```

Run the container:

```bash
docker run --rm trading-app
```

## Running Locally (without Docker)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip3 install -r requirements.txt
```

Then run:

```bash
python scrapers/djia/download_djia.py
```

Or:

```bash
python3 scrapers/djia/download_djia.py
```

## Requirements

- Python 3.x
- yfinance (see requirements.txt)
- Docker (for containerized execution)
