---
trigger: model_decision
description: Structure - EEG BIDS Derivatives, Modular Pipelines y Results I/O
---


## Project Structure & Organization

```
<repo>/
├── BIDS/                          # BIDS compliant data (raw & derivatives)
├── Preprocessing_pipeline_new/    # EEG Preprocessing modules
├── ERPs_new/                      # ERP analysis modules
├── Statistics/                    # Statistical analysis (LMMs, etc.)
├── junifer_markers/               # Feature extraction pipeline (spectral/conn)
├── mw_classification_pipeline/    # Machine Learning and classification
├── Behavior/                      # Behavioral analysis and dashboards
├── utils/                         # Shared project utilities
├── results/                       # Generated outputs (gitignored)
└── tests/                         # Automated tests
```

### Rules
- **BIDS Derivatives**: Preprocessed EEG data must be stored in `BIDS/derivatives/sub-{subject}/eeg/`.
- **Feature Storage**: Extracted markers and features belong in `BIDS/features/sub-{subject}/eeg/`.
- **Results Isolation**: All pipeline outputs (plots, CSVs, models) MUST go to `results/{pipeline_name}/`. NEVER write results inside source code directories.
- **Source Layout**: Complex pipelines (e.g., classification) should use a `utils/` subdirectory for internal logic.
- **Triad Development**: For new modular features, create:
    1. Implementation (e.g., `new_analysis.py`)
    2. Unit Tests (in `tests/`)
    3. Documentation (Docstrings in the script)

### Checklist
- [ ] Data paths follow the `BIDS/` structure.
- [ ] Results are isolated in the `results/` directory.
- [ ] No hardcoded absolute paths; use `utils/bids_compliance.py` or similar helpers.