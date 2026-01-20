.PHONY: help scraper visualize clean

# Default target
help:
	@echo "Trading Project - Convenience commands:"
	@echo ""
	@echo "  make scraper    - Run DJIA scraper (delegates to scrapers/djia/Makefile)"
	@echo "  make visualize  - Run visualization (delegates to visualizations/djia/Makefile)"
	@echo "  make clean      - Clean up all Docker images"
	@echo ""
	@echo "For detailed usage, see individual app README files:"
	@echo "  - scrapers/djia/README.md"
	@echo "  - visualizations/djia/README.md"
	@echo ""
	@echo "Or run 'make help' in each app directory for app-specific commands."

# Scraper - delegate to app's Makefile
scraper:
	cd scrapers/djia && $(MAKE) run

# Visualization - delegate to app's Makefile
# Usage: make visualize ARGS="--granularity daily --column Close"
visualize:
	cd visualizations/djia && $(MAKE) run ARGS="$(ARGS)"

# Clean up all Docker images
clean:
	cd scrapers/djia && $(MAKE) clean || true
	cd visualizations/djia && $(MAKE) clean || true
