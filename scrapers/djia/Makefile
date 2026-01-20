.PHONY: help build run clean

# Default target
help:
	@echo "DJIA Scraper - Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Run the scraper"
	@echo "  make clean  - Remove Docker image"

# Build the Docker image
build:
	docker build -t djia-scraper .

# Run the scraper
run: build
	docker run --rm -v $$(pwd):/app djia-scraper

# Clean up Docker image
clean:
	docker rmi djia-scraper 2>/dev/null || true
