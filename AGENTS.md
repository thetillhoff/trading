# Agent Guidelines

## General Rules

- Don't write .md files unless asked to.
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

**IMPORTANT**: Always consult `DEVELOPMENT.md` first for architecture, design decisions, and implementation details. See that file for data flow, module responsibilities, and extension points.

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

### Documentation & Git

- **Keep documentation up to date**: Update README files and Makefiles when making changes.
- **No historical name references**: Do not keep references to old names (files, methods, variables, etc.) in documentation. Update all references to current names immediately.
- **Git commit at stable milestones**: When development reaches a stable milestone (completed feature, fixed issues, significant improvements), **persist the milestone via git commit** with a descriptive message. Remind the use if you think they forgot.

### Data & File Management

- **Data persistence**: CSV files should be used for caching downloaded data to avoid re-downloading on each run.
- **File naming**: Files should have descriptive, meaningful names that clearly indicate their purpose.

### Conflict Resolution

- If there's a conflict between user instructions and documentation, ask the user to validate before proceeding.
