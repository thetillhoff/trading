# Development Journey & Project Structure

## Project Overview

This is a Python trading data project that downloads and analyzes DJIA (Dow Jones Industrial Average)
historical data using the yfinance library. The project is containerized with Docker for easy deployment
and reproducibility.

## AI Assistant Guidelines

**IMPORTANT**: When working on this project, AI assistants should:

1. **Always consult DEVELOPMENT.md first** - Read this file to understand project structure, design
   decisions, and development history before making changes.

2. **Follow established patterns** - Adhere to the design decisions and patterns documented in this file.
   If a proposed change conflicts with existing decisions, **ask the user for validation** before
   proceeding.

3. **Update DEVELOPMENT.md** - When making significant changes or adding new features, update this file
   to reflect:
   - New development phases
   - Updated design decisions
   - Changes to project structure
   - Important user instructions or preferences

4. **Preserve user preferences** - Document important user instructions and preferences in this file so
   they can be referenced in future AI chat sessions when history is unavailable. **This is critical for
   new AI agents that don't have historical knowledge** - they must be able to understand all important
   rules and conventions by reading this file.

5. **No historical name references** - Do not keep references to old names (files, methods, variables,
   etc.) in documentation. Update all references to current names immediately. Only mention old names in
   examples showing what NOT to do (e.g., bad naming examples).

6. **Git commit at stable milestones** - When development reaches a stable milestone (completed feature,
   fixed issues, significant improvements), **persist the milestone via git commit** with a descriptive
   message and push to the remote repository. This ensures progress is saved and the repository stays up
   to date. Stable milestones should be committed immediately, not deferred.

### Important User Instructions

**CRITICAL**: These instructions must be followed by all AI agents, especially new ones without
historical context. All important rules and preferences must be documented here.

- **Docker-first approach**: Always use Docker containers instead of local Python installation. The user
  prefers `docker compose` (v2 syntax) over `docker-compose`.
- **Makefile usage**: Prefer Makefile commands (`make up`, `make run`, etc.) for common operations.
- **Data persistence**: CSV files should be used for caching downloaded data to avoid re-downloading on
  each run.
- **File naming**: Files should have descriptive, meaningful names that clearly indicate their purpose
  (e.g., `download_djia.py` instead of generic names like `hello.py`).
- **No historical name references**: Do not keep references to old names (files, methods, variables,
  etc.) in documentation. Update all references to current names immediately. Only mention old names in
  examples showing what NOT to do.
- **Git commit at stable milestones**: When development reaches a stable milestone (completed feature,
  fixed issues, significant improvements), **persist the milestone via git commit** with a descriptive
  message and push to the remote repository. Ensure all relevant changes are committed, not just the
  most recent ones. Stable milestones should be committed immediately to preserve progress.
- **Keep documentation up to date**: Keep usage instructions in README files and Makefiles up to date
  when making changes. If you add new features, update the relevant README with examples. If you add
  new commands, update the Makefile and its help text.
- **Follow software engineering best practices**: Apply good software engineering practices throughout
  development. For example:
  - If a file becomes too long (e.g., >500 lines), consider splitting it into smaller modules
  - Extract complex methods into separate functions or classes when they become unwieldy
  - Maintain clear separation of concerns between modules
  - Keep functions focused on a single responsibility
  - Refactor code when it becomes difficult to understand or maintain
- **Restructure when necessary**: Restructuring of files, folders, and code can happen whenever necessary
  and useful. Don't hesitate to reorganize the codebase to improve maintainability, clarity, or
  structure. This includes:
  - Moving files to more appropriate locations
  - Splitting large modules into smaller, focused modules
  - Reorganizing directory structures
  - Refactoring code to better align with project patterns
  - Updating imports and references after restructuring
- **Document important rules**: Any important rules, preferences, or conventions mentioned during
  development should be added to this section so future AI agents can discover them.
- **Standardize CLI arguments**: All visualization and analysis scripts should use consistent argument
  names, formats, defaults, and help text. Common arguments like `--start-date`, `--end-date`,
  `--column`, `--min-confidence`, and `--min-wave-size` must work the same way across all scripts.
  This ensures users can apply the same parameters across different tools without confusion.
- **Conflict resolution**: If there's a conflict between user instructions and DEVELOPMENT.md, ask the
  user to validate before proceeding.

## Project Structure

```text
trading/
├── scrapers/             # Scraper subprojects
│   └── djia/            # DJIA scraper
│       ├── download_djia.py  # Downloads and caches DJIA data
│       ├── Dockerfile         # Scraper-specific Docker configuration
│       ├── requirements.txt   # Scraper-specific dependencies
│       ├── Makefile           # Scraper-specific build/run commands
│       ├── README.md          # Scraper usage documentation
│       └── djia_data.csv      # Cached data (generated)
├── visualizations/       # Visualization subprojects
│   └── djia/            # DJIA visualizations
│       ├── data_loader.py    # Loads data from scraper outputs
│       ├── data_processor.py # Processes and aggregates data
│       ├── visualizer.py     # Creates charts and graphs
│       ├── visualize_djia.py # Main CLI script
│       ├── Dockerfile         # Visualization-specific Docker configuration
│       ├── requirements.txt   # Visualization-specific dependencies
│       ├── Makefile           # Visualization-specific build/run commands
│       └── README.md          # Visualization usage documentation
├── Makefile              # Convenience commands (delegates to app Makefiles)
├── .dockerignore         # Files to exclude from Docker build context
├── .gitignore           # Files to exclude from version control
├── README.md            # Project overview documentation
└── DEVELOPMENT.md       # This file - development history and structure
```

### Key Files

- **scrapers/djia/download_djia.py**: DJIA scraper that downloads data from 1900 onwards, caches it in
  CSV format, and displays summary statistics
- **scrapers/djia/Dockerfile**: Independent Docker configuration for the scraper
- **scrapers/djia/requirements.txt**: Scraper-specific dependencies (yfinance, pandas)
- **visualizations/djia/visualize_djia.py**: Main CLI for generating DJIA visualizations with
  customizable parameters
- **visualizations/djia/data_loader.py**: Flexible data loader that can read from any scraper output
- **visualizations/djia/data_processor.py**: Processes data with granularity
  (daily/weekly/monthly/yearly) and aggregation (mean/max/min/etc.)
- **visualizations/djia/visualizer.py**: Extensible visualization engine for creating charts
- **visualizations/djia/Dockerfile**: Independent Docker configuration for visualizations
- **visualizations/djia/requirements.txt**: Visualization-specific dependencies (pandas, matplotlib)
- **Makefile**: Provides convenient shortcuts for running each app independently

## Development Journey

### Phase 1: Initial Setup

- Started as a basic Python project
- Created main application script for downloading trading data
- Added basic README with usage instructions

### Phase 2: Trading Data Integration

- Integrated yfinance library to download financial data
- Modified application to download DJIA (^DJI) historical data starting from 1900-01-01
- Added data display functionality showing the first few rows of the dataset

### Phase 3: Docker Containerization

- Created `Dockerfile` using Python 3.11-slim for a lightweight container
- Set up `docker-compose.yml` for easier container management
- Added volume mounting to persist data between container runs
- Created `.dockerignore` to optimize build context
- Updated README with Docker usage instructions

### Phase 4: Developer Experience Improvements

- Created `Makefile` with common commands (build, run, up, down, clean, rebuild)
- Standardized on `docker compose` (v2 syntax) instead of `docker-compose`
- Removed version field from docker-compose.yml (optional in newer versions)

### Phase 5: Data Persistence & Optimization

- Implemented CSV caching to avoid re-downloading data on every run
- Added logic to check for existing `djia_data.csv` file
- If CSV exists, load from file; otherwise download and save
- Added pandas dependency for CSV handling
- Created `.gitignore` to exclude generated CSV files from version control
- Enhanced output with data shape and date range information

### Phase 6: Project Reorganization

- Reorganized project structure to accommodate multiple scraper subprojects
- Created `scrapers/` directory with `djia/` subdirectory
- Moved DJIA scraper to `scrapers/djia/download_djia.py`
- Updated CSV file path to be relative to script location
- Updated Dockerfile and docker-compose.yml to work with new structure
- Established pattern for future scraper subprojects

### Phase 7: Visualization System

- Created `visualizations/` directory following the same pattern as `scrapers/`
- Implemented modular architecture with separate components:
  - `data_loader.py`: Flexible data loading from scraper outputs
  - `data_processor.py`: Time series processing with granularity and aggregation
  - `visualizer.py`: Extensible chart generation engine
  - `visualize_djia.py`: CLI interface for generating visualizations
- Added support for:
  - Time range filtering (start/end dates)
  - Granularity selection (daily, weekly, monthly, yearly)
  - Aggregation methods (mean, max, min, median, sum, first, last)
  - Column selection (Close, High, Low, Open, Volume)
- Added matplotlib dependency for chart generation
- Updated Makefile with visualization command
- Designed for extensibility: easy to add new chart types, data sources, and visualizations

### Phase 8: Independent App Architecture

- Refactored to independent Docker containers for each app
- Each scraper has its own `Dockerfile` and `requirements.txt`
- Each visualization has its own `Dockerfile` and `requirements.txt`
- Removed root-level Docker files (Dockerfile, docker-compose.yml, requirements.txt)
- Updated Makefile to support independent execution:
  - `make scraper-build` and `make scraper-run` for scrapers
  - `make visualize-build` and `make visualize` for visualizations
- Apps can now run independently:
  - Scrapers can run on schedule (e.g., daily cron jobs)
  - Visualizations can run on-demand when requested
- Data loader updated to find scraper data in multiple locations (local dev and Docker)

### Phase 9: Self-Contained App Documentation

- Each app now has its own `README.md` with comprehensive usage documentation
- Each app has its own `Makefile` with app-specific commands
- Root Makefile delegates to app Makefiles for convenience
- Apps are now fully self-contained and can be used independently:
  - Users can navigate to an app directory and use it without AI assistance
  - Clear documentation for manual usage scenarios
  - Each app's Makefile provides help with `make help`
- Documentation includes:
  - Overview and features
  - Requirements
  - Usage examples (Docker and local)
  - Command-line arguments (for visualizations)
  - Troubleshooting guides
  - Scheduling examples (for scrapers)

## Key Design Decisions

### 1. Docker-First Approach

**Decision**: Use Docker containers instead of local Python installation

**Rationale**:

- Ensures consistent environment across different machines
- Isolates dependencies
- Makes deployment easier
- Reproducible builds

### 2. Data Caching Strategy

**Decision**: Save downloaded data to CSV file

**Rationale**:

- Reduces API calls to yfinance
- Faster subsequent runs
- Works offline after initial download
- CSV format is human-readable and portable

### 3. Makefile for Common Operations

**Decision**: Create Makefile instead of relying on Docker commands directly

**Rationale**:

- Simplifies common workflows
- Provides consistent interface
- Reduces typing and potential errors
- Self-documenting with `make help`

### 4. Volume Mounting in Docker Compose

**Decision**: Mount current directory as volume in docker-compose.yml

**Rationale**:

- Allows CSV file to persist on host machine
- Enables code changes without rebuilding image
- Data survives container removal

### 5. Python 3.11-slim Base Image

**Decision**: Use slim variant of Python Docker image

**Rationale**:

- Smaller image size
- Faster builds and pulls
- Still includes all necessary Python functionality

### 6. Scraper Subproject Structure

**Decision**: Organize scrapers into `scrapers/` directory with individual subdirectories

**Rationale**:

- Allows for multiple independent scraper projects
- Each scraper can have its own data files
- Clear separation of concerns
- Easy to add new scrapers without affecting existing ones
- Data files are stored alongside their respective scrapers

### 7. Visualization Subproject Structure

**Decision**: Organize visualizations into `visualizations/` directory mirroring `scrapers/` structure

**Rationale**:

- Consistent project organization pattern
- Each visualization subproject can target specific scrapers
- Clear separation between data collection and visualization
- Easy to add new visualization types without affecting existing ones

### 8. Modular Visualization Architecture

**Decision**: Separate visualization into loader, processor, and visualizer components

**Rationale**:

- Single Responsibility Principle: each component has one clear purpose
- Extensibility: easy to add new data sources, processing methods, or chart types
- Testability: components can be tested independently
- Reusability: components can be used across different visualization subprojects
- Future-proof: new features can be added without breaking existing functionality

### 9. CLI-Based Visualization Interface

**Decision**: Use command-line arguments for visualization parameters

**Rationale**:

- Flexible: supports automation and scripting
- Docker-friendly: works well in containerized environments
- Extensible: easy to add new parameters without breaking existing usage
- Clear: explicit parameters make the interface self-documenting

### 10. Independent App Architecture

**Decision**: Each app (scraper, visualization) has its own Dockerfile and requirements.txt

**Rationale**:

- Independence: apps can run separately without affecting each other
- Scheduling: scrapers can run on cron/schedule, visualizations on-demand
- Dependency isolation: each app only includes what it needs
- Scalability: easy to add new apps without modifying existing ones
- Maintenance: changes to one app don't require rebuilding others
- Deployment: apps can be deployed independently to different environments

### 11. Self-Contained App Documentation

**Decision**: Each app has its own README.md and Makefile

**Rationale**:

- Usability: users can use apps without AI assistance or project-wide knowledge
- Independence: each app is fully documented and can be used standalone
- Clarity: app-specific documentation is more focused and easier to find
- Maintainability: documentation lives with the code it describes
- Discoverability: `make help` in each app directory shows available commands
- Manual usage: supports scenarios where users need to run apps manually

### 12. Standardized CLI Arguments

**Decision**: All visualization and analysis scripts use consistent argument names, formats, defaults,
and help text

**Rationale**:

- User experience: users can apply the same parameters across different tools without confusion
- Consistency: reduces cognitive load when switching between scripts
- Predictability: arguments behave the same way everywhere
- Maintainability: easier to document and support when arguments are standardized

**Standard Arguments**:

- `--start-date`: Start date (inclusive) in format YYYY-MM-DD (optional)
- `--end-date`: End date (inclusive) in format YYYY-MM-DD (optional)
- `--column`: Column to analyze/visualize (default: Close). Available: Close, High, Low, Open, Volume
- `--min-confidence`: Minimum confidence (0.0-1.0) for wave detection (default: 0.6)
- `--min-wave-size`: Minimum wave size as ratio of price range (default: 0.05 = 5%)
- `--granularity`: Time granularity (where applicable): daily, weekly, monthly, yearly (default: daily)
- `--aggregation`: Aggregation method (where applicable): mean, max, min, median, sum, first, last
  (default: mean)

**Implementation**: All scripts that accept these arguments must use the exact same parameter names,
formats, defaults, and help text. When adding new scripts, follow this standard.

## Current State

The project is functional and ready for use. It:

- ✅ Downloads DJIA historical data from yfinance
- ✅ Caches data in CSV format for faster subsequent runs
- ✅ Runs in Docker containers with proper isolation
- ✅ Provides convenient Makefile commands
- ✅ Generates visualizations with customizable parameters:
  - Time range filtering
  - Granularity selection (daily, weekly, monthly, yearly)
  - Aggregation methods (mean, max, min, median, sum, first, last)
  - Column selection (Close, High, Low, Open, Volume)
- ✅ Elliott Wave pattern detection with color-coded visualization
- ✅ Trading signal detection (buy/sell) with target prices and stop-loss calculation
- ✅ Trade evaluation and backtesting with performance metrics
- ✅ Filter optimization for Elliott Wave detection
- ✅ Multi-chart generation (combined or separate visualizations)
- ✅ Buy-and-hold comparison for trade evaluation
- ✅ Hold-through-stop-loss strategy testing
- ✅ Includes comprehensive documentation (EXAMPLES.md, module READMEs)

## Future Considerations

Potential enhancements:

- Add data update logic (check if new data is available)
- Implement data analysis/visualization features
- Add support for multiple tickers
- Create data validation/quality checks
- Add configuration file for customizable parameters
- Implement logging instead of print statements
- Add unit tests
- Create data export to other formats (JSON, Parquet)

## Dependencies

### Scraper Dependencies (scrapers/djia/requirements.txt)

- **yfinance** (>=0.2.0): Yahoo Finance data downloader
- **pandas** (>=2.0.0): Data manipulation and CSV handling

### Visualization Dependencies

- **visualizations/djia/requirements.txt**:
  - pandas (>=2.0.0): Data manipulation and CSV handling
  - matplotlib (>=3.7.0): Chart and graph generation
- **visualizations/elliott_wave_optimizer/requirements.txt**:
  - pandas (>=2.0.0): Data manipulation
  - numpy (>=1.24.0): Numerical computations
- **visualizations/trading_signals/requirements.txt**:
  - pandas (>=2.0.0): Data manipulation
  - matplotlib (>=3.7.0): Chart generation
- **visualizations/trade_evaluator/requirements.txt**:
  - pandas (>=2.0.0): Data manipulation
  - numpy (>=1.24.0): Numerical computations
  - matplotlib (>=3.7.0): Chart generation

## Usage Summary

### Quick Start (Root Level)

```bash
# Run DJIA scraper
make scraper

# Generate visualization
make visualize ARGS="--granularity daily --column Close"
```

### App-Specific Usage

Each app can be used independently from its own directory:

```bash
# Scraper usage
cd scrapers/djia
make help      # Show available commands
make build     # Build Docker image
make run       # Run the scraper

# Visualization usage
cd visualizations/djia
make help      # Show available commands and examples
make build     # Build Docker image
make run ARGS="--granularity daily --column Close"
```

For detailed usage, see:

- `scrapers/djia/README.md` - Scraper documentation
- `visualizations/djia/README.md` - Visualization documentation

The scraper will automatically download data on first run and use cached CSV on subsequent runs.
Visualizations are saved as PNG files in the project root directory (or specified output directory).

Each app runs independently:

- **Scrapers**: Can be scheduled (e.g., daily cron job) to keep data updated
- **Visualizations**: Run on-demand when you need charts
