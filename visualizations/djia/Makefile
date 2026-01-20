.PHONY: help build run clean

# Default target
help:
	@echo "DJIA Visualization - Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Run visualization (use ARGS for parameters)"
	@echo "  make clean  - Remove Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS=\"--granularity daily --column Close\""
	@echo "  make run ARGS=\"--start-date 2023-01-01 --end-date 2023-12-31 --granularity monthly --aggregation mean --column Close\""
	@echo "  make run ARGS=\"--granularity weekly --aggregation max --column High\""

# Build the Docker image
build:
	docker build -t djia-visualizer .

# Run visualization
# ARGS should contain visualization parameters
run: build
	@if [ -z "$(ARGS)" ]; then \
		echo "Error: ARGS parameter required. Example: make run ARGS=\"--granularity daily --column Close\""; \
		exit 1; \
	fi
	docker run --rm \
		-v $$(pwd)/../../scrapers:/app/scrapers:ro \
		-v $$(pwd)/../../:/app/project_root \
		djia-visualizer python visualize_djia.py $(ARGS) --output-dir /app/project_root

# Clean up Docker image
clean:
	docker rmi djia-visualizer 2>/dev/null || true
