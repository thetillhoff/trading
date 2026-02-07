# Agent Guidelines

## General Rules

- Don't write .md files unless asked to.
- Don't add make targets unless asked to.
- DO NOT GIVE ME HIGH LEVEL SHIT, IF I ASK FOR FIX OR EXPLANATION, I WANT ACTUAL CODE OR EXPLANATION!!! I DON'T WANT "Here's how you can blablabla"
- Be casual unless otherwise specified
- Be terse
- Suggest solutions that I didn't think about—anticipate my needs
- Treat me as an expert
- Be accurate and thorough
- Give the answer immediately. Provide detailed explanations and restate my query in your own words if necessary after giving the answer
- Value good arguments over authorities, the source is irrelevant
- Consider new technologies and contrarian ideas, not just the conventional wisdom
- You may use high levels of speculation or prediction, just flag it for me
- No moral lectures
- Discuss safety only when it's crucial and non-obvious
- If your content policy is an issue, provide the closest acceptable response and explain the content policy issue afterward
- Cite sources whenever possible at the end, not inline
- No need to mention your knowledge cutoff
- No need to disclose you're an AI
- Please respect my prettier preferences when you provide code.
- Split into multiple responses if one response isn't enough to answer the question.
- Follow the repository pattern

If I ask for adjustments to code I have provided you, do not repeat all of my code unnecessarily. Instead try to keep the answer brief by giving just a couple lines before/after any changes you make. Multiple code blocks are ok.

## Project-Specific Guidelines

**IMPORTANT**: Always consult `DEVELOPMENT.md` first for architecture, design decisions, and implementation details. See that file for data flow, module responsibilities, and extension points. Keep it up to date, too. At the same time ensure there are no unnecessary contents, like code examples. It should only describe the general architecture and design decisions, not the specific code.

### Docker & Makefile Usage

- **Docker-first approach**: Always use Docker containers instead of local Python installation. Use `docker compose` (v2 syntax), not `docker-compose`.
- **Never run Python directly**: Do NOT run `python` or `python3` commands directly on the host machine. Always use Docker containers via `make` commands or `docker compose run --rm cli`.
- **Makefile usage**: Prefer Makefile commands (`make evaluate`, `make grid-search`, etc.) for common operations. See `make help` for available commands. This is especially important for the cli agent.

### Code Quality & Structure

- **Restructure when necessary**: Don't hesitate to reorganize the codebase to improve maintainability, clarity, or structure.
- **File length**: If a file becomes too long (>500 lines), consider splitting it into smaller modules.
- **Extract complex methods**: Break down complex methods into separate functions or classes when they become unwieldy.
- **Separation of concerns**: Maintain clear separation between modules.
- **Single responsibility**: Keep functions focused on a single responsibility.
- **DRY (Don't Repeat Yourself)**: Try to avoid repeating the same code or logic in multiple places.
- **KISS (Keep it simple stupid)**: Try to keep things simple, and don't overcomplicate. This applies to the architecture, too. If something becomes complex, there's probably a redesign needed.
- **Single source of truth**: Keep a single source of truth, both for configuration and documentation like `ROADMAP.md` & `HYPOTHESIS_TEST_RESULTS.md`.
- **Automated testing**: Write tests for all code. 80% coverage is the goal. Run those tests automatically while developing new features and after making changes to validate the code. If you find something that should be tested, but isn't yet, add tests & run them without being asked.

### Documentation & Git

- **Keep documentation up to date**: Update README files and Makefiles when making changes.
- **No historical name references**: Do not keep references to old names (files, methods, variables, etc.) in documentation. Update all references to current names immediately.
- **Git commit at stable milestones**: When development reaches a stable milestone (completed feature, fixed issues, significant improvements), **persist the milestone via git commit** with a descriptive message. Remind the use if you think they forgot.

### Hypothesis Test Results (HYPOTHESIS_TEST_RESULTS.md)

- **Scientific hypothesis style**: Each entry states a **hypothesis** (what is being tested), then **findings** (what was observed), then a **conclusion** (accepted/rejected/modified). Do not write in narrative form (“we ran tests…”, “latest run”, “next steps”).
- **No operational context**: Do not reference run dates, result folders, or “test date / generated / configs tested” in the doc. It should read as a catalogue of hypotheses and evidence, not a lab log.
- **Optimality claims**: If a conclusion states that “X is optimal”, always state **under which circumstances** (e.g. instrument, period, baseline config, objective). Example: “Under circumstances: walk-forward on DJIA, full period 2000–2020, EW+all indicators, position 0.35 — risk_reward 2.5 is optimal.”
- **New hypotheses**: A new hypothesis is created the moment the configs for it are created.
- **Auto cleanup**: If the results of a hypothesis were analyzed, the configs and result files can be deleted.
- **Always keep baseline up to date**: The `config/baseline.yaml` should always be the best performing config, and the one that is used by the `make evaluate` command. Ensure it always contains the best performing config.

### Roadmap (ROADMAP.md)

- **Priority order**: The roadmap contains a prioritized list of next steps to improve the product. Even if there's just an idea or suggestion, it should be added there.
- **Describe what, not how**: Each item states *what is being tested* or *what to build* (e.g. “Indicator params: whether EMA/MACD/RSI period choices beat baseline”). Do not use process instructions like “Run grid”, “add findings”, or “document in HYPOTHESIS_TEST_RESULTS”.
- **In progress → hypothesis doc, not roadmap**: If a hypothesis is *currently being tested* (configs run, results pending), it belongs in HYPOTHESIS_TEST_RESULTS.md. Do not list it on the roadmap; the roadmap holds only work that is not yet started or is the next step after current work.
- **Completed items**: If an item is completed, it should be removed from the roadmap.
- **Unordered list**: The roadmap should be an unordered list, except for priority sections. That way, changes to the file are minimal when adding new items.
- **No instructions**: The roadmap should not contain instructions on how to run tests or use the code. It only describes next steps to improve the product, in the tersest form possible without losing the gist.
- **No current status**: The status quo is not a "roadmap" item, so it should not be described in it.

### Data & File Management

- **Data persistence**: CSV files should be used for caching downloaded data to avoid re-downloading on each run.
- **File naming**: Files should have descriptive, meaningful names that clearly indicate their purpose.

### Conflict Resolution

- If there's a conflict between user instructions and documentation, ask the user to validate before proceeding.
