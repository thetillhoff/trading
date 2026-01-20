.PHONY: help build run clean

# Default target
help:
	@echo "Elliott Wave Filter Optimizer - Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Run optimizer (use ARGS for parameters)"
	@echo "  make clean  - Remove Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS=\"--target-waves 10\""
	@echo "  make run ARGS=\"--granularity monthly --verbose\""
	@echo "  make run ARGS=\"--test-filters 'confidence=0.7,size=0.1'\""

# Build the Docker image
# Using BuildKit for better caching (speeds up pip installs)
build:
	DOCKER_BUILDKIT=1 docker build -t elliott-wave-optimizer .

# Run optimizer
# ARGS should contain optimizer parameters
run: build
	docker run --rm \
		-v $$(pwd)/../../scrapers:/app/scrapers:ro \
		-v $$(pwd)/../djia:/app/djia:ro \
		elliott-wave-optimizer python optimize_filters.py $(ARGS)

# Clean up Docker image
clean:
	docker rmi elliott-wave-optimizer 2>/dev/null || true
