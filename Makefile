.PHONY: help scraper visualize multi-charts optimize-filters trading-signals evaluate-trades clean

# Default target
help:
	@echo "Trading Project - Convenience commands:"
	@echo ""
	@echo "  make scraper         - Run DJIA scraper (delegates to scrapers/djia/Makefile)"
	@echo "  make visualize       - Run visualization (delegates to visualizations/djia/Makefile)"
	@echo "  make multi-charts    - Generate multiple charts at once (delegates to visualizations/djia/Makefile)"
	@echo "  make optimize-filters - Optimize Elliott Wave filters (delegates to visualizations/elliott_wave_optimizer/Makefile)"
	@echo "  make trading-signals - Analyze trading signals (delegates to visualizations/trading_signals/Makefile)"
	@echo "  make evaluate-trades - Evaluate trading signal performance (delegates to visualizations/trade_evaluator/Makefile)"
	@echo "  make clean           - Clean up all Docker images"
	@echo ""
	@echo "For detailed usage, see individual app README files:"
	@echo "  - scrapers/djia/README.md"
	@echo "  - visualizations/djia/README.md"
	@echo "  - visualizations/elliott_wave_optimizer/README.md"
	@echo "  - visualizations/trading_signals/README.md"
	@echo "  - visualizations/trade_evaluator/README.md"
	@echo ""
	@echo "Or run 'make help' in each app directory for app-specific commands."

# Scraper - delegate to app's Makefile
scraper:
	cd scrapers/djia && $(MAKE) run

# Visualization - delegate to app's Makefile
# Usage: make visualize ARGS="--granularity daily --column Close"
visualize:
	cd visualizations/djia && $(MAKE) run ARGS="$(ARGS)"

# Multi-charts - delegate to app's Makefile
# Usage: make multi-charts ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"
multi-charts:
	cd visualizations/djia && $(MAKE) multi-charts ARGS="$(ARGS)"

# Filter optimizer - delegate to app's Makefile
# Usage: make optimize-filters ARGS="--target-waves 10"
optimize-filters:
	cd visualizations/elliott_wave_optimizer && $(MAKE) run ARGS="$(ARGS)"

# Trading signals - delegate to app's Makefile
# Usage: make trading-signals ARGS="--column Close --signal-type buy"
trading-signals:
	cd visualizations/trading_signals && $(MAKE) run ARGS="$(ARGS)"

# Trade evaluator - delegate to app's Makefile
# Usage: make evaluate-trades ARGS="--column Close"
evaluate-trades:
	cd visualizations/trade_evaluator && $(MAKE) run ARGS="$(ARGS)"

# Clean up all Docker images
clean:
	cd scrapers/djia && $(MAKE) clean || true
	cd visualizations/djia && $(MAKE) clean || true
	cd visualizations/elliott_wave_optimizer && $(MAKE) clean || true
	cd visualizations/trading_signals && $(MAKE) clean || true
	cd visualizations/trade_evaluator && $(MAKE) clean || true
