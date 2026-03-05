---
trigger: model_decision
description: Style - Code Quality, Numpy Docstrings y Naming semántico
---

## Code Style & Quality

### Rules
- **Scientific Documentation**: Use Numpy/Scipy style for all docstrings. Every function and class must be documented.
- **Sectioning**: Use clear visual separators (e.g., `# ===`) to group logically related parts of a script (Imports, Configuration, Helper Functions, Main).
- **Naming Semantics**: Use domain-specific names (e.g., `high_onoff_probes`, `marker_intensity`). Avoid generic names like `df1`, `data_list`.
- **Type Hints**: All function signatures MUST include type hints for parameters and return values.
- **No Magic Numbers**: Extract all constants and scientific parameters to the top of the script or, preferably, to `config.yaml`.
- **Transparency & Exclusions**: 
    - If data points or participants are excluded, the code must allow for (and the results must report) results both with and without these exclusions.
    - Every data cleaning step must be explicitly documented in the code and results log.
- **No try/except**: Do not use `try/except` in scientific scripts. Errors must surface immediately with a clear message. Use guard clauses and assertions instead.
- **Explicit Seeds**: Any stochastic step must use `random_state` from config, never implicit defaults.

### Checklist
- [ ] All functions have descriptive docstrings.
- [ ] Code is separated into clear logical blocks.
- [ ] Type hints are present on all new/modified functions.
- [ ] Variables use snake_case and are descriptive.
- [ ] No `try/except` blocks present.
- [ ] All stochastic operations have an explicit `random_state`.