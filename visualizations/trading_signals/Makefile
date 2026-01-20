.PHONY: help build run clean

# Default target
help:
	@echo "Trading Signals Analyzer - Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Run signal analysis (use ARGS for parameters)"
	@echo "  make clean  - Remove Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS=\"--column Close\""
	@echo "  make run ARGS=\"--start-date 2020-01-01 --end-date 2024-12-31 --column Close\""
	@echo "  make run ARGS=\"--column Close --signal-type buy --risk-reward 3.0\""

# Build the Docker image
# Using BuildKit for better caching (speeds up pip installs)
build:
	DOCKER_BUILDKIT=1 docker build -t trading-signals-analyzer .

# Run signal analysis
# ARGS should contain analysis parameters
run: build
	docker run --rm \
		-v $$(pwd)/../../scrapers:/app/scrapers:ro \
		-v $$(pwd)/../djia:/app/djia:ro \
		-v $$(pwd)/../../:/app/project_root \
		trading-signals-analyzer python analyze_signals.py $(ARGS) --output-dir /app/project_root

# Clean up Docker image
clean:
	docker rmi trading-signals-analyzer 2>/dev/null || true
