# Python Trading Project

A Python project that downloads and displays DJIA (Dow Jones Industrial Average) data using yfinance.

## Quick Start

The easiest way to run the project is using the Makefile:

```bash
make up
```

Or see all available commands:

```bash
make help
```

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
