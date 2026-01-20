.PHONY: help build run clean

# Default target
help:
	@echo "Trade Evaluator - Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Evaluate trades (use ARGS for parameters)"
	@echo "  make clean  - Remove Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS=\"--column Close\""
	@echo "  make run ARGS=\"--start-date 2015-01-01 --end-date 2024-12-31 --column Close\""
	@echo "  make run ARGS=\"--column Close --signal-type buy --max-days 30\""

# Build the Docker image
build:
	DOCKER_BUILDKIT=1 docker build -t trade-evaluator .

# Run trade evaluation
# ARGS should contain evaluation parameters
run: build
	docker run --rm \
		-v $$(pwd)/../../scrapers:/app/scrapers:ro \
		-v $$(pwd)/../djia:/app/djia:ro \
		-v $$(pwd)/../trading_signals:/app/trading_signals:ro \
		-v $$(pwd)/../../:/app/project_root \
		trade-evaluator python evaluate_trades.py $(ARGS) --output-dir /app/project_root

# Clean up Docker image
clean:
	docker rmi trade-evaluator 2>/dev/null || true
