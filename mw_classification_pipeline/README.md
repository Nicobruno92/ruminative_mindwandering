# MW Classification Pipeline

Individual-level decodability of mind-wandering (MW) from EEG markers — **Section 3 of Paper 2**.
Tests whether the group-level neural signatures found by CBPT (Section 2) are strong enough to
predict MW state trial-by-trial, per probe dimension.

> CBPT establishes *which* dimensions leave a detectable group-level neural trace.
> Classification asks whether that trace is strong enough to predict MW state in a specific person.
> The decodability hierarchy mirrors CBPT signal density — validating both methods.

**Headline results** (`results/MW_Classification/`):
`onoff` (AUC≈0.703, 16/29 subjects sig.) > `valence` (0.658) > `confidence` (0.630) ≈
`selfother` (0.632) > `time` (0.593 ≈ chance). LOSO (onoff): AUC≈0.628 < within=0.703 — the gap
quantifies the idiosyncratic component of MW signatures.

---

## Two cross-validation regimes

The same ML engine (`utils/`) is driven by two pipelines that differ only in how folds are formed:

| Pipeline | Question | CV scheme | Entry point |
|----------|----------|-----------|-------------|
| [`within_subject_pipeline/`](within_subject_pipeline/) | Can we decode MW *within a single person*? | Per-subject train/test (median split per dimension) | `run_within_subject_classification.py` |
| [`loso_pipeline/`](loso_pipeline/) | Does a MW signature *generalize across people*? | Leave-One-Subject-Out | `run_loso_classification.py` |

Each has its own `config.yaml`, `run_local.sh`/`run_cluster.sh`, `tests/`, `type1_error/`, and a
`spatial_decoding/` searchlight variant. See their READMEs for details.

`cross_decoding/` is a **post-hoc** analysis: it reuses the per-dimension predictions both pipelines
already save and builds a generalization matrix (no retraining). Off-diagonals are largely explained
by label correlation — do **not** read neural specificity into them (see memory
`cross-decoding-is-label-correlation`).

---

## Directory map

```
mw_classification_pipeline/
├── utils/                       # Shared engine (imported by BOTH pipelines via sys.path)
│   ├── data_utils.py            #   load Junifer probe CSVs, build label contrasts, family filter
│   ├── ml_utils.py              #   model build (rf/xgb/lr/ocsvm/iforest), CV, SMOTE, metrics, SHAP
│   ├── analysis_utils.py        #   n_runs distribution + permutation null (within & LOSO)
│   ├── spatial_decoding_utils.py#   per-electrode searchlight + max-stat FWER + topomaps
│   ├── cross_decoding_utils.py  #   post-hoc generalization matrix from saved predictions
│   ├── plotting_utils.py        #   subject densities, SHAP boxplots, AUC-vs-X diagnostics
│   ├── simulation_utils.py      #   synthetic null matrices for Type-I error calibration
│   └── logging_utils.py         #   AnalysisLogger
├── within_subject_pipeline/     # Within-subject CV  (see its README)
├── loso_pipeline/               # LOSO CV            (see its README)
├── cross_decoding/              # Post-hoc cross-decoding matrix
├── scripts/                     # Cross-pipeline figure/plot generators
│   ├── generate_combined_classification_figure.py   # main paper figure (within + LOSO)
│   ├── generate_spatial_panel.py / *_spatial_*      # spatial topomap panels
│   ├── plot_auc_vs_median_position.py               # AUC vs split-position diagnostic
│   └── run_all_plots.py / .sh                        # regenerate every figure
├── tests/                       # Cross-pipeline tests (spatial, median-position plot)
└── results/MW_Classification/   # Outputs (gitignored)
    ├── WithinSubject/<contrast>/<family>/<model>/
    ├── LOSO/<contrast>/<family>/<model>/
    ├── SpatialDecoding/{WithinSubject,LOSO}/
    ├── cross_decoding/<scheme>_<model>_<family>/
    └── data_cache/              # precomputed feature matrices (speeds reruns)
```

---

## Core concepts (shared by both pipelines)

**Data source.** Junifer aggregated per-probe markers, one CSV per `(subject, task, probe, epoch_type)`
under `data_paths.features_root`. Only pre-probe bins (`-5..-1`) are used. Subjects `"02".."43"`,
tasks `Sart1..Sart4` (see root `CLAUDE.md`).

**Label contrasts** (`label_contrasts` in config). A probe dimension (`onoff`, `valence`,
`selfother`, `time`, `confidence`) is binarized — usually `within_subject_median` with a `gap`
neutral zone. `restrict_to` enables `_offtask` variants (classify content only within MW probes).
Scale convention: `0 = off-task/negative/past/self/low-confidence`, `100 = the opposite`.

**Feature families** (`feature_families` in config). Subsets of markers run as independent jobs:
`all`, `erp` (P1/N1/P2/P3a/P3b), `sleep` (slow waves, spindles), `spectral` (band power),
`connectivity` (wSMI), `information_theory` (PE, Kolmogorov complexity). Each family declares which
`epoch_types` to load (`state`/`sleep`/`evoked`) and column `prefixes`.

**Statistics.** `n_runs` seeded passes build the true-AUC distribution; `permutation_runs`
label-shuffles build the null; p = (#perm ≥ true + 1)/(N + 1). MCC handled per analysis.
See memory `loso-pipeline-decisions-2026-06` for confidence-weighting / imputation / p-value choices.

---

## Quick start

```bash
conda activate ML        # ML tasks use the ML env (see CLAUDE.md)

# Within-subject, all families, local sequential run
cd within_subject_pipeline && bash run_local.sh config.yaml

# LOSO, local
cd ../loso_pipeline && bash run_local.sh config.yaml

# Cluster (SLURM): submit arrays, then merge
bash run_cluster.sh config.yaml          # submits true + perm arrays
bash run_merge.sh                         # LOSO only: aggregate when jobs finish

# Cross-pipeline figures
cd ../scripts && python run_all_plots.py
```

Smoke test before any real run: `--dry_run` loads data and prints shapes without classifying.

---

## Conventions (enforced — see root `CLAUDE.md`)

- **No hardcoded paths/params** — everything flows from `config.yaml`; each run writes `used_config.yaml`.
- **No `try/except`** in scientific code — let errors surface.
- **Explicit `random_state`** on every stochastic step.
- **Results isolation** — outputs only under `results/`, never in source dirs.
- **Color palette** — load `color_palette.yaml` (repo root); dimension = hue, significance = fill
  (solid = sig, hollow = n.s.). Never hardcode hex.
- **Envs** — `ML` for classification, `plots` for figure-only scripts, `eeg`/`junifer` for upstream features.

---

## How to extend the pipeline

1. **New probe dimension / contrast** → add a key under `label_contrasts` + `dimensions`
   (or `run_contrasts`) in *both* configs. `restrict_to` for `_offtask` variants.
2. **New marker family** → add to `feature_families` with `epoch_types` + `prefixes`; add the name to
   `run_families`. Word-boundary matching means `P1` matches `Pz_P1` but not `Pz_P10`.
3. **New model** → extend `build_model` in `utils/ml_utils.py` and the `--model_type` choices.
4. **New diagnostic plot** → add to `utils/plotting_utils.py`, call from the pipeline's plotting block,
   register in `scripts/run_all_plots.py`. Honor the palette/significance convention.
5. **Always** add a test in the relevant `tests/` and run `type1_error/` calibration when changing the
   statistics. Keep the SLURM array math (`array_idx = subject_idx*4 + task_idx`) in sync with the
   Python entrypoint.

See each sub-pipeline's README for its run modes, SLURM layout, and outstanding TODOs.
