# Within-Subject Pipeline

**Within-subject** classification of MW dimensions from EEG markers — tests whether MW state can be
decoded **trial-by-trial inside a single person**. This is the primary decodability result of
Section 3: the AUC hierarchy across dimensions (`onoff` 0.703 > `valence` 0.658 > `confidence` 0.630
≈ `selfother` 0.632 > `time` 0.593) mirrors CBPT signal density. ~45% of subjects are not individually
decodable even for `onoff` — a result, not a failure.

See the [pipeline README](../README.md) for shared concepts (data source, contrasts, families,
statistics). This file covers what is specific to the within-subject regime.

---

## What it does

For each `(contrast × family × model)` combination, **independently per subject**:

1. Build the binary label per dimension (usually within-subject median split with a `gap` neutral zone).
2. Train/test within that subject's probes; repeat over `n_runs` seeds → per-subject AUC distribution.
3. Label-shuffle `permutation_runs` times → per-subject null; combine for a group-level test.
4. Save per-subject metrics, densities, SHAP, and `*_consolidated_sample_predictions.csv`
   (consumed by `cross_decoding/`).

Output: `results/MW_Classification/WithinSubject/<contrast>/<family>/<model>/` + `used_config.yaml`.

---

## Files

```
within_subject_pipeline/
├── config.yaml                          # ALL parameters (paths, subjects, contrasts, dimensions, families, n_runs, slurm)
├── run_within_subject_classification.py # Main entry point
├── precompute_data_cache.py             # One-off: cache feature matrices (~5 min load → <1 s reads)
├── run_local.sh                         # Sequential local run over all combinations
├── run_cluster.sh                       # Submit SLURM arrays: 1 job per true run + 1 per permutation
├── run_cluster_worker.sh                # Per-job worker (--run_idx / --perm_idx)
├── run_plots.sh                         # Regenerate plots only
├── spatial_decoding/                    # Per-electrode searchlight (own README)
├── type1_error/                         # Type-I error calibration on synthetic null matrices
└── tests/                               # pytest: analysis, missing-features, restrict_to, performance
```

---

## Running

```bash
conda activate ML

# Smoke test — load data, print shapes, no classification
python run_within_subject_classification.py --config config.yaml --dry_run

# (recommended) cache feature matrices once — every job then reads in <1 s
python precompute_data_cache.py --config config.yaml

# Local sequential (all combinations in config)
bash run_local.sh config.yaml
bash run_local.sh config.yaml --skip-permutation     # faster, no null

# Single override
python run_within_subject_classification.py --config config.yaml \
    --contrast valence_within_median --family spectral --model_type rf
```

### Cluster (SLURM)

Like LOSO, the cluster path splits true runs and permutations into separate array jobs:

```bash
python precompute_data_cache.py --config config.yaml   # do this first
bash run_cluster.sh config.yaml                         # submits <name>_true and <name>_perm arrays
squeue -u $USER
```

`run_cluster_worker.sh` calls the entry point with `--run_idx R` (true, `--skip_permutation`) or
`--perm_idx P` (perm), each writing a partial output.

> ⚠️ **Known gap / TODO.** `run_cluster.sh` references a `merge_ws_results.py` aggregator that does
> **not yet exist** in this directory (LOSO has `merge_loso_results.py`). If you split runs/perms
> across the cluster, you currently have no automated merge step. **Either** run `run_local.sh`
> (sequential, self-aggregating) for the published numbers, **or** port `../loso_pipeline/merge_loso_results.py`
> to the `WithinSubject/` layout before relying on cluster output. This is the first thing to build
> when scaling the within-subject pipeline up.

---

## Key config knobs (`config.yaml`)

| Block | What to set |
|-------|-------------|
| `data_paths` | `features_root`, `results_root` |
| `subjects` / `tasks` | cohort; `min_samples`, `min_minority_ratio` per-subject filters |
| `run_contrasts` + `dimensions` | which dimensions; `split_method`, `gap`, `positive_class_name`, `restrict_to` for `_offtask` |
| `feature_families` + `run_families` | marker subsets (`epoch_types` + `prefixes`) |
| `classifiers.run_models` | `rf` (default), `xgb`, `lr`, `ocsvm`, `iforest` |
| `n_runs`, `n_permutations` | distribution / null sizes |
| oversampling block | SMOTE family, `within` scope, `smote_k_neighbors` |
| `*_n_jobs` / `n_perm_jobs` | parallelism (keep within `cpus_per_task`; watch loky oversubscription) |
| `slurm` | partition, cpus_per_task, conda_env |

Per-subject filtering is stricter here than in LOSO — subjects below `min_samples` or
`min_minority_ratio` are dropped *for that contrast* and reported in the subject-exclusion summary
(transparency requirement). Always report numbers with and without exclusions.

---

## Validate before trusting results

```bash
cd within_subject_pipeline
conda run -n ML pytest tests/ -q
bash type1_error/submit_type1_error_within.sh     # Type-I error calibration on synthetic null
```

`test_restrict_to_contrast.py` and `test_missing_features.py` guard the two trickiest behaviors
(`_offtask` restriction, sparse marker columns). Re-run `type1_error/` after any statistics change.

---

## Continuing development

- **Build `merge_ws_results.py`** (top priority for cluster use) — port the LOSO merger to the
  `WithinSubject/<contrast>/<family>/<model>/` path and update the `run_cluster.sh` header.
- **Add a dimension/family/model** → edit `config.yaml` only (see pipeline README "How to extend").
- **Change the statistics** → update `utils/analysis_utils.py` (within-subject functions) +
  `utils/ml_utils.py`, add a test, re-run `type1_error/`.
- **Spatial searchlight** → see `spatial_decoding/README.md`.
- **Speed** → always `precompute_data_cache.py` first; the bottleneck is CSV loading, not the ML.
- Honor anti-forking-paths: `_offtask` and confidence variants are **exploratory** — flag them as such
  and report sensitivity, don't promote a post-hoc winner to a confirmatory claim.
