---
applyTo: "**"
---
## Documentation & Maintenance Rules

### 1. Code-Level Documentation
- **Mandatory Docstrings**: Every function, class, and module must have a Numpy/Scipy style docstring.
- **Explain "Why", not "What"**: Inline comments should explain the reasoning behind complex logic or scientific decisions, not restate what the code does.
- **Entrypoint Documentation**: Main scripts (e.g., `run_*.py`) must include a module-level docstring with usage examples and a description of the pipeline.

### 2. Project Maintenance
- **README Updates**: Any change that modifies how to run the project, its dependencies, or the data structure must be updated in the main `README.md`.
- **Environment Documentation**: When adding new libraries, immediately update `environment.yml` or `pyproject.toml`. Document the reason if it's a non-standard scientific library.
- **Changelog**: Maintain a record of major scientific or architectural changes to ensure provenance and reproducibility.

### 3. Development Lifecycle
- **Refactoring vs. Features**: Explicitly separate refactoring tasks from new feature development in commits and documentation.
- **On-Boarding Focus**: Write documentation and code such that another researcher (or your future self) can understand the pipeline logic within 10 minutes.
- **Legacy Cleanup**: When modifying a file, remove obsolete comments and "TODOs" that are no longer relevant.

### Checklist
- [ ] Module-level docstrings exist for new scripts.
- [ ] `README.md` or `environment.yml` updated if necessary.
- [ ] Comments focus on scientific rationale (the "Why").
- [ ] Code is self-documenting and maintainable.
