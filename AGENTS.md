# daily_for_cursor

Small, dependency-free demos for practicing Cursor. Treat this repo as a living example of Cursor setup: `AGENTS.md` for project facts, `.cursor/rules/` for scoped conventions, `.cursor/skills/` for repeatable workflows.

## Commands

- Demo: `python3 -m beamsearch`
- Tests: `python3 -m unittest discover -s tests -v`

## Conventions

- Python 3.9+, standard library only unless a sample is specifically about a third-party library.
- Prefer type hints, dataclasses, and short modules over frameworks.
- Keep each sample self-contained: code under `<name>/`, tests under `tests/test_<name>.py`, and a short README section.
- Do not add package managers, Docker, or CI unless the user asks.

## Workflow

1. Match existing samples (`beamsearch/`) before inventing a new layout.
2. Add or update unit tests with the change.
3. Run the unittest command above and keep it green.
4. Update `README.md` when adding a sample or changing how something is run.

Human-facing Cursor guidance lives in [docs/cursor-best-practices.md](docs/cursor-best-practices.md).
