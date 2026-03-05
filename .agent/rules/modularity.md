---
trigger: always_on
description: Modularity - YAML Config, Pure Functions y Pipelines modulares
---

# modularity.md

## Modularity & Design

### Rules
- **YAML-Only Configuration**: **Everything** configurable must live in a YAML file.
    - NEVER hardcode parameters in Python or Bash.
    - AVOID double-configurations (redundant keys in different files).
    - PREVENT overwriting configs inside Python/Bash logic unless explicitly requested per-run.
- **Modular Pipelines**: Full pipelines must have their own `utils/`, `tests/`, and sub-modules if they are complex. Simple scripts can remain single-file.
- **Pure Functions**: Prefer functions that take inputs and return outputs without side effects (no modifying global state).
- **I/O Isolation**: Keep data loading/saving separate from computation logic to enable unit testing.
- **Data-Code Separation**: 
    - Maintain a strict boundary between raw data (immutable), processing scripts (versioned), and derivatives (versioned).
    - Never store code inside data directories or vice versa.
    - Path resolution must be handled by a centralized utility (e.g., `utils/bids_compliance.py`).

### Checklist
- [ ] No circular imports.
- [ ] Complex pipelines have a dedicated `utils/` or directory structure.
- [ ] NO hardcoded parameters; all values come from `config.yaml`.
- [ ] No redundant configurations across files.
- [ ] Raw data remains untouched (Read-Only).