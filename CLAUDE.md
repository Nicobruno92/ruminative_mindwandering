# EEG Mind-Wandering Analysis — Claude Instructions



Virtual Environment Use. You never have to create a new one, there is an specific environment designed for each kind of task. You have to activate the appropriate one before running.

basic stuff => base
crating plots specific tasks => plots
anything related to machine learning => ML
anything related to natural languange processing => nlp
for anything related to eeg analysis => eeg
for junifer specific tasks => junifer

## Critical Rules (Always Apply)

1. **NEVER hardcode paths** → use `utils/bids_compliance.py`
2. **NEVER hardcode parameters** → load from `config.yaml`
3. **Subject IDs**: zero-padded strings `"02"` to `"43"` (no `"01"`)
4. **Tasks**: `["Sart1", "Sart2", "Sart3", "Sart4"]` (case-sensitive)
5. **Pre-probe analysis**: filter distance `-5` to `-1`
6. **No `try/except` blocks** in scientific scripts — let errors surface; fix root causes
7. **No magic numbers** — all constants go to `config.yaml` or top of script
8. **Explicit random seeds** — always use `random_state` from config, never implicit defaults

## Essential Imports

```python
import yaml
from utils.bids_compliance import read_epochs, save_evokeds, load_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

config = yaml.safe_load(open('config.yaml'))
```

## Data Locations

```
_RAW_DATA/{sub}/                          → Raw BrainVision
BIDS/raw/{sub}/eeg/                       → BIDS raw
BIDS/derivatives/{sub}/eeg/               → Preprocessed (*_desc-autoPreproc_epo.fif)
BIDS/features/{sub}/eeg/
  ├── junifer/                            → Spectral/connectivity (*_markers.pkl)
  └── mne_evokeds/                        → ERPs (*_desc-probe-{N}_{label}_ave.fif)
results/{pipeline}/                       → Outputs (gitignored)
```

## Project Structure

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

## Config Locations

```
Preprocessing_pipeline_new/config.yaml    → ICA, artifacts
ERPs_new/config.yaml                      → ROIs, baseline
Statistics/config.yaml                    → LMM, thresholds
junifer_markers/*/config.yaml             → Markers, bands
```

## Standard Workflow

```python
# Load & filter
epochs = read_epochs("04", "Sart1", config['project']['derivatives_root'])
epochs = filter_epochs_by_distance_to_probe(epochs, distance=5)  # -5 to -1

# Classify on/off-task
classified = classify_onoff_epochs(epochs, split='median')
evoked_on  = classified['high'].average()
evoked_off = classified['low'].average()

# Save
save_evokeds([evoked_on, evoked_off], "04", "Sart1",
             config['project']['features_root'],
             desc="probe-015", labels=['onTask', 'offTask'])
```

## Event Structure

Format: `go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`

- **Dimensions**: 0–100 scale (0=off-task, 100=on-task)
- **Distance**: negative=before probe, positive=after
- **Use**: `-5` to `-1` for mental state prediction

## Run Commands

```bash
# Single subject
python Preprocessing_pipeline_new/preprocessing_pipeline.py \
  --config config.yaml --subject 04 --task Sart1

python ERPs_new/make_probe_evokeds.py \
  --config config.yaml --subject 04 --task Sart1

# Batch (SLURM)
sbatch Preprocessing_pipeline_new/run_preprocessing_slurm.sh
sbatch ERPs_new/run_complete_erp_pipeline.sh

# Junifer (sequential)
cd junifer_markers/1.markers_h5_creation && sbatch slurm_array_junifer.sh
cd ../2.h5_to_pkl && sbatch batch_convert_h5_to_pkl_parallel.sh
cd ../3.aggregate_probes && sbatch run_aggregate_slurm.sh
```

## SLURM Array Index

`array_idx = subject_idx * 4 + task_idx`

Example: Sub-04 (idx=2), Sart2 (idx=1) → `2*4+1 = 9`

## Key Functions

```python
# I/O
read_epochs(subject, task, derivatives_root)
save_evokeds(evokeds, subject, task, features_root, desc, labels)
load_evokeds(subject, task, features_root, desc, labels)

# Analysis
filter_epochs_by_distance_to_probe(epochs, distance)  # keeps -N to -1
classify_onoff_epochs(epochs, split='median')          # {'high': on, 'low': off}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check zero-padding (`"04"`), case (`"Sart1"`), BIDS functions |
| No epochs match | Verify distance filter, check `*_desc-autoPreproc_epo.fif` exists |
| Memory error | Add `gc.collect()`, use `max_files_per_batch` parameter |
| Missing config param | Check correct config file, use `config['project']['key']` path |

---

## Scientific Guardrails

- **Context-First**: Before generating code, read `README.md` to understand the project structure. Do not guess file paths.
- **Epistemic Humility**: Flag high-risk statistical decisions (thresholding, p-values) and ask for scientific justification. Do not default to `p < 0.05` without context.
- **Hallucination Check**: Only import libraries listed in `pyproject.toml` or `environment.yml`. If a new library is needed, propose adding it to the environment file first.
- **BIDS Compliance**: Never write to `data/raw`. Validate entity ordering in filenames. When creating `.tsv`/`.csv` output, suggest a corresponding `.json` sidecar.
- **Scientific Self-Evaluation**: Before declaring a task finished, review generated plots/stats for physiological plausibility (ERP latencies, frequency band power distributions).
- **Determinism**: Scientific pipelines must be fully deterministic. No conditional fallbacks. Optional steps controlled by config flags, not exception catching.
- **Garden of Forking Paths**: If exploring multiple parameters or thresholds, document all of them and report result sensitivity. Flag exploratory findings as *exploratory*, not *confirmatory*.
- **Anti-HARKing**: Do not rewrite the goal of a script to match what the data showed. Always distinguish planned analysis from post-hoc discoveries.

---

## Code Style

- **Docstrings**: Numpy/Scipy style. Every function, class, and module must be documented.
- **Sectioning**: Use `# ===` separators to group Imports / Configuration / Helper Functions / Main.
- **Naming**: Use domain-specific names (`high_onoff_probes`, `marker_intensity`). Avoid `df1`, `data_list`.
- **Type hints**: Required on all new/modified function signatures.
- **Transparency**: If participants/trials are excluded, provide results both with and without exclusions. Document every cleaning step.

---

## Modularity

- **YAML-only config**: Never hardcode parameters in Python or Bash. No double-configurations.
- **Pure Functions**: Prefer functions with no side effects; keep I/O separate from computation.
- **Data-Code Separation**: Raw data is immutable. Path resolution via `utils/bids_compliance.py`.
- **Structure for new features**: Implementation + unit tests in `tests/` + docstrings.
- **Results isolation**: All outputs (plots, CSVs, models) go to `results/{pipeline_name}/`. Never write results into source code directories.

---

## Git Practices

- **Atomic Scientism**: Do not mix scientific changes with cosmetic changes in the same commit.
- **The "Why" Mandate**: Commit messages for parameter/logic changes must explain the scientific rationale.
- **No `git checkout`** to solve development issues or revert changes — use normal edits and commits.
- **Large files**: Never commit files > 10MB. Use DataLad or `.gitignore`.
- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `exp:` (for experiments).

---

## Automation Checklist

Before declaring any pipeline stage complete:
- [ ] Fast local test script exists and passes
- [ ] Config loaded with `yaml.safe_load()`
- [ ] Subject is string: `"04"` not `4`
- [ ] Task case-correct: `"Sart1"` not `"sart1"`
- [ ] BIDS functions used (no hardcoded paths)
- [ ] Pre-probe filter applied for mental state analysis
- [ ] Results folder contains `used_config.yaml`
- [ ] Performance metrics reviewed and documented
- [ ] SLURM scripts match Python entrypoint
- [ ] No binary/large files in `git status`
- [ ] All stochastic operations have explicit `random_state`
- [ ] No `try/except` blocks present
- [ ] Type hints on all new/modified functions
