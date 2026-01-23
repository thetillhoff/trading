# Development Guidelines

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
   to reflect new design decisions or important user instructions.

4. **Preserve user preferences** - Document important user instructions and preferences in this file so
   they can be referenced in future AI chat sessions. **This is critical for new AI agents that don't
   have historical knowledge**.

### Important User Instructions

**CRITICAL**: These instructions must be followed by all AI agents.

- **Docker-first approach**: Always use Docker containers instead of local Python installation. The user
  prefers `docker compose` (v2 syntax) over `docker-compose`.
- **Never run Python directly**: Do NOT run `python` or `python3` commands directly on the host machine.
  Always use Docker containers via `make run` or `docker run`. This ensures consistent environments and
  avoids dependency issues.
- **Makefile usage**: Prefer Makefile commands (`make up`, `make run`, etc.) for common operations.
- **Data persistence**: CSV files should be used for caching downloaded data to avoid re-downloading on
  each run.
- **File naming**: Files should have descriptive, meaningful names that clearly indicate their purpose.
- **No historical name references**: Do not keep references to old names (files, methods, variables,
  etc.) in documentation. Update all references to current names immediately.
- **Git commit at stable milestones**: When development reaches a stable milestone (completed feature,
  fixed issues, significant improvements), **persist the milestone via git commit** with a descriptive
  message and push to the remote repository. Stable milestones should be committed immediately.
- **Keep documentation up to date**: Keep usage instructions in README files and Makefiles up to date
  when making changes.
- **Follow software engineering best practices**: Apply good practices throughout development:
  - If a file becomes too long (e.g., >500 lines), consider splitting it into smaller modules
  - Extract complex methods into separate functions or classes when they become unwieldy
  - Maintain clear separation of concerns between modules
  - Keep functions focused on a single responsibility
- **Restructure when necessary**: Don't hesitate to reorganize the codebase to improve maintainability,
  clarity, or structure.
- **Standardize CLI arguments**: All visualization and analysis scripts should use consistent argument
  names, formats, defaults, and help text.
- **Conflict resolution**: If there's a conflict between user instructions and DEVELOPMENT.md, ask the
  user to validate before proceeding.

## Key Design Decisions

### 1. Docker-First Approach

Use Docker containers instead of local Python installation for consistent environments, isolated
dependencies, and reproducible builds.

### 2. Data Caching Strategy

Save downloaded data to CSV files to reduce API calls, enable faster subsequent runs, and allow
offline operation after initial download.

### 3. Makefile for Common Operations

Use Makefiles instead of raw Docker commands for simplified workflows, consistent interface, and
self-documenting commands via `make help`.

### 4. Independent App Architecture

Each app (scraper, visualization) has its own Dockerfile and requirements.txt for independence,
scheduling flexibility, dependency isolation, and maintainability.

### 5. Self-Contained App Documentation

Each app has its own README.md and Makefile for standalone usability without AI assistance or
project-wide knowledge.

### 6. Modular Visualization Architecture

Separate visualization into loader, processor, and visualizer components for single responsibility,
extensibility, testability, and reusability.

### 7. Standardized CLI Arguments

All scripts use consistent argument names, formats, defaults, and help text:

- `--start-date`: Start date (inclusive) in format YYYY-MM-DD (optional)
- `--end-date`: End date (inclusive) in format YYYY-MM-DD (optional)
- `--column`: Column to analyze/visualize (default: Close). Available: Close, High, Low, Open, Volume
- `--min-confidence`: Minimum confidence (0.0-1.0) for wave detection (default: 0.6)
- `--min-wave-size`: Minimum wave size as ratio of price range (default: 0.05 = 5%)
- `--granularity`: Time granularity: daily, weekly, monthly, yearly (default: daily)
- `--aggregation`: Aggregation method: mean, max, min, median, sum, first, last (default: mean)
