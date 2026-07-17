# LOSO Pipeline

**Leave-One-Subject-Out** classification of MW dimensions from EEG markers — tests whether a MW
neural signature **generalizes across people**. Complements the within-subject pipeline; the
within-vs-LOSO AUC gap (0.703 → 0.628 for `onoff`) quantifies the idiosyncratic component of MW
signatures.

See the [pipeline README](../README.md) for shared concepts (data source, contrasts, families,
statistics). This file covers what is specific to LOSO.

---

## What it does

For each `(contrast × family × model)` combination:

1. Pool all subjects' pre-probe markers, build the binary label via the contrast.
2. Run `n_runs` seeded LOSO passes (each subject held out once per pass) → true-AUC distribution.
3. Run `permutation_runs` label-shuffled passes (shuffle within subject) → null distribution.
4. p = (#perm ≥ true + 1) / (N + 1); aggregate per-subject metrics, SHAP, predictions, plots.

Output: `results/MW_Classification/LOSO/<contrast>/<family>/<model>/`, plus
`*_consolidated_sample_predictions.csv` (consumed by `cross_decoding/`) and `used_config.yaml`.

---

## Files

```
loso_pipeline/
├── config.yaml                     # ALL parameters (paths, subjects, contrasts, families, n_runs, SMOTE, n_jobs, slurm)
├── run_loso_classification.py      # Main entry point
├── merge_loso_results.py           # Aggregate parallel SLURM job outputs into consolidated files
├── run_local.sh                    # Sequential local run over all combinations
├── run_cluster.sh                  # Submit SLURM arrays: 1 job per true run + 1 per permutation
├── run_cluster_worker.sh           # Per-job worker (--run_idx / --perm_idx)
├── run_merge.sh                    # Convenience wrapper for merge_loso_results.py
├── run_plots.sh                    # Regenerate plots only
├── spatial_decoding/               # Per-electrode searchlight (own README) — max-stat FWER topomaps
├── type1_error/                    # Type-I error calibration on synthetic null matrices
└── tests/                          # pytest: data, ML, analysis, confidence weighting, cross-decoding, perf
```

---

## Running

```bash
conda activate ML

# Smoke test — load data, print shapes, no classification
python run_loso_classification.py --config config.yaml --dry_run

# Local sequential (all combinations in config)
bash run_local.sh config.yaml
bash run_local.sh config.yaml --skip-permutation   # faster, no null

# Single override
python run_loso_classification.py --config config.yaml --contrast ON_vs_OFF_within_median --family spectral
```

### Cluster (SLURM)

The cluster path **splits true runs and permutations into separate array jobs** so every
run/perm executes on its own node, then merges:

```bash
bash run_cluster.sh config.yaml     # submits <name>_true (0..n_runs-1) and <name>_perm (0..permutation_runs-1)
squeue -u $USER                      # wait for all jobs
bash run_merge.sh                    # == python merge_loso_results.py --config config.yaml
```

`run_cluster_worker.sh` calls the entry point with `--run_idx R` (true) or `--perm_idx P` (perm),
each writing a partial file; `merge_loso_results.py` recombines them into the same consolidated
outputs a sequential run would produce and computes the final p-values. **You must run the merge** —
nothing aggregates automatically.

---

## Key config knobs (`config.yaml`)

| Block | What to set |
|-------|-------------|
| `data_paths` | `features_root` (Junifer features), `results_root` |
| `subjects` / `tasks` | cohort (`"02".."43"`, `Sart1..Sart4`); `min_samples`, `min_minority_ratio` filters |
| `label_contrasts` + `run_contrasts` | which dimensions; `split_method`, `gap`, `restrict_to` for `_offtask` |
| `feature_families` + `run_families` | marker subsets; `epoch_types` + `prefixes` |
| `model_type` / `run_models` | `rf` (default), `xgb`, `lr`, `ocsvm`, `iforest` |
| `n_runs`, `permutation_runs` | distribution / null sizes (defaults 100 / 500) |
| `oversampling_method`, `oversampling_scope`, `smote_k_neighbors` | SMOTE family, `within` vs `global` |
| `confidence_weight` | per-contrast; **off by default** (see memory `loso-pipeline-decisions-2026-06`) |
| `*_n_jobs`, `perm_cv_n_jobs`, `true_cv_n_jobs`, `lmm_n_jobs` | parallelism; keep folds×lmm ≤ cpus_per_task |
| `slurm` | partition, cpus_per_task, conda_env, chunking |

**Threading gotcha** (documented in config): total threads per perm ≈ `perm_cv_n_jobs × lmm_n_jobs`;
keep ≤ `cpus_per_task` or you oversubscribe loky workers (see memory `conda-envs-and-test-runner`).

---

## Validate before trusting results

```bash
cd loso_pipeline
conda run -n ML pytest tests/ -q                       # unit + pipeline tests
bash type1_error/submit_type1_error_loso.sh            # Type-I error calibration (synthetic null)
```

`type1_error/` fits the full pipeline on label-free synthetic matrices that preserve covariance and
subject membership; the empirical FPR should sit at the nominal α. Re-run it whenever you touch the
statistics (split method, weighting, p-value, n_jobs).

---

## Continuing development

- **Add a contrast/family/model** → edit `config.yaml` only (see pipeline README "How to extend").
- **Change the statistics** → update `utils/analysis_utils.py` + `utils/ml_utils.py`, add a test,
  re-run `type1_error/`.
- **Spatial searchlight** → see `spatial_decoding/README.md`; it reuses this engine per electrode with
  a max-statistic FWER null.
- **Migrating old outputs** → `scripts/migrate_loso_subject_metrics.py` upgrades legacy per-subject files.
- **Open items**: confidence-weighting is exploratory for content dimensions (justified mainly for
  `onoff`); always run WITH and WITHOUT and report sensitivity (anti-forking-paths). The `all` family
  intentionally omits some marker types — see memory `loso-all-family-excludes-state`.
