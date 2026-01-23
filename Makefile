.PHONY: help scraper scraper-list visualize multi-charts optimize-filters trading-signals evaluate-trades backtest backtest-grid clean download evaluate grid-search params timeline

# Default target
help:
	@echo "Trading System - Unified CLI"
	@echo ""
	@echo "Core Commands:"
	@echo "  make download              Download market data for instruments"
	@echo "  make evaluate              Evaluate a trading strategy with charts"
	@echo "  make grid-search           Compare multiple strategy configurations"
	@echo "  make params                Show all configurable parameters"
	@echo "  make timeline              Generate trade timeline from CSV"
	@echo ""
	@echo "Command Options (append to any command):"
	@echo "  ARGS='...'                 Pass CLI arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make download ARGS='--list'                    # List available instruments"
	@echo "  make download ARGS='djia sp500'                # Download specific data"
	@echo "  make evaluate ARGS='--instrument djia --charts' # Evaluate with visualizations"
	@echo "  make evaluate ARGS='--max-timeline-trades 50'  # Limit timeline trades"
	@echo "  make grid-search ARGS='--instrument sp500'      # Compare strategies"
	@echo "  make timeline ARGS='trades.csv'                # Visualize from CSV"
	@echo ""
	@echo "Data Directory:"
	@echo "  ./data/                    Downloaded market data (created by download command)"
	@echo ""
	@echo "Output:"
	@echo "  Charts saved to current directory"
	@echo "  CSV files for trade data export"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 Remove unused Docker images"

# Pass PRODUCTION flag to sub-make
ifdef PRODUCTION
  PROD_FLAG = PRODUCTION=1
else
  PROD_FLAG =
endif

# Unified CLI (runs in Docker)
DOCKER_CLI = docker compose run --rm --build cli python -u -m

download:
	$(DOCKER_CLI) cli.download $(ARGS)

evaluate:
	$(DOCKER_CLI) cli.evaluate $(ARGS)

grid-search:
	$(DOCKER_CLI) cli.grid_search $(ARGS)

params:
	$(DOCKER_CLI) cli.params

timeline:
	$(DOCKER_CLI) cli.timeline $(ARGS)

# Cleanup
clean:
	docker system prune -f || true
