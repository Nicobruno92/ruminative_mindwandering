# MW Classification Pipeline

Individual-level decodability of mind-wandering (MW) from EEG markers — **Section 3 of Paper 2**.
Tests whether the group-level neural signatures found by CBPT (Section 2) are strong enough to
predict MW state trial-by-trial, per probe dimension.

> CBPT establishes *which* dimensions leave a detectable group-level neural trace.
> Classification asks whether that trace is strong enough to predict MW state in a specific person.
> The decodability hierarchy mirrors CBPT signal density — validating both methods.

**⚠ Headline numbers below are provisional.** `loso_pipeline/config.yaml` and
`within_subject_pipeline/config.yaml` currently have **uncommitted local changes**
(`git status` on this branch), and re-verifying the numbers below against the results
currently on disk turned up a real, substantial drift from what was previously documented
here (order-of-magnitude changes for `confidence`/`selfother`, and LOSO now covers all five
dimensions rather than just `onoff`). See [§12 Current results snapshot](#12-current-results-snapshot)
for the full, re-derived numbers and an explicit discussion of the drift — **do not cite the numbers
in this paragraph, or any number from before this rewrite, without re-checking §12 first.**

Within-subject decodability (group level, recomputed 2026-07-27 directly from the raw per-run/
per-permutation CSVs, same statistic as the manuscript figure):
`onoff` (AUC≈0.710) > `valence` (0.662) > `confidence` (0.652) > `selfother` (0.596) >
`time` (0.582, weakest but still FDR-significant). LOSO (same recomputation): `confidence`
(0.612) and `onoff` (0.589) generalize across subjects; `valence`/`selfother`/`time` sit at
chance after FDR correction. See §12 for the full tables, caveats, and how to reproduce them.

---

## Two cross-validation regimes

The same ML engine (`utils/`) is driven by two pipelines that differ only in how folds are formed:

| Pipeline | Question | CV scheme | Entry point |
|----------|----------|-----------|-------------|
| [`within_subject_pipeline/`](within_subject_pipeline/) | Can we decode MW *within a single person*? | Per-subject train/test (median split per dimension) | `run_within_subject_classification.py` |
| [`loso_pipeline/`](loso_pipeline/) | Does a MW signature *generalize across people*? | Leave-One-Subject-Out | `run_loso_classification.py` |

Each has its own `config.yaml`, `run_local.sh`/`run_cluster.sh`, `tests/`, `type1_error/`, and a
`spatial_decoding/` searchlight variant. See their READMEs for day-to-day usage; **§ Detailed
Methods below is the authoritative, code-verified description of what both pipelines actually do**
(the sub-READMEs are accurate for CLI usage but were not re-verified line-by-line against the code
in this pass).

`cross_decoding/` is a **post-hoc** analysis: it reuses the per-dimension predictions both pipelines
already save and builds a generalization matrix (no retraining). Off-diagonals are largely explained
by label correlation — do **not** read neural specificity into them (see memory
`cross-decoding-is-label-correlation`, and §11 below).

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
├── cross_decoding/               # Post-hoc cross-decoding matrix
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
    ├── Type1Error/{WithinSubject,LOSO}/
    ├── cross_decoding/<scheme>_<model>_<family>/
    └── data_cache/              # precomputed feature matrices (speeds reruns)
```

---

## Core concepts (shared by both pipelines)

**Data source.** Junifer aggregated per-probe markers, one CSV per `(subject, task, probe, epoch_type)`
under `data_paths.features_root`. The pre-probe-only restriction (`-5..-1`, see root `CLAUDE.md`) is
enforced **upstream**, by the Junifer aggregation step (`junifer_markers/`) — it is not re-applied
anywhere in `utils/data_utils.py`; if you ever point this pipeline at a differently-aggregated CSV
set, re-verify that restriction still holds. Subjects `"02".."43"`, tasks `Sart1..Sart4`.

**Label contrasts** (`label_contrasts` in config). A probe dimension (`onoff`, `valence`,
`selfother`, `time`, `confidence`) is binarized — usually `within_subject_median` with a `gap`
neutral zone. `restrict_to` enables `_offtask` variants (classify content only within MW probes).
Scale convention: `0 = off-task/negative/past/self/low-confidence`, `100 = the opposite`. Full
algorithm in §4 below.

**Feature families** (`feature_families` in config). Subsets of markers run as independent jobs:
`all`, `erp` (P1/N1/P2/P3a/P3b), `sleep` (slow waves, spindles), `spectral` (band power),
`connectivity` (wSMI), `information_theory` (PE, Kolmogorov complexity). Each family declares which
`epoch_types` to load (`state`/`sleep`/`evoked`) and column `prefixes`, matched by **plain substring
containment** (`utils/data_utils.py`) — despite the name, this is not a word-boundary/regex match;
with the shipped ERP prefixes (`P1`, `N1`, `P3a`, `P3b`) that distinction never bites in practice, but
don't assume `P1` can't also match a hypothetical `P10` column if you add one. Only `all` is in
`run_families` for both pipelines — `erp`/`sleep`/`spectral`/`connectivity`/`information_theory` are
configured but not part of the currently-run analysis.

**Statistics.** `n_runs` seeded passes build the true-AUC distribution; `permutation_runs`
label-shuffles build the null; see §8 for the exact (and, it turns out, not fully unified) p-value
formulas and the multiple-comparisons scheme. See memory `loso-pipeline-decisions-2026-06` for
confidence-weighting / imputation / p-value background.

---

## Quick start

```bash
conda activate ML        # ML tasks use the ML env (see CLAUDE.md)

# Within-subject: cache feature matrices once (saves ~5 min/job -> <1s), then run
cd within_subject_pipeline
python precompute_data_cache.py --config config.yaml
bash run_local.sh config.yaml

# LOSO, local
cd ../loso_pipeline && bash run_local.sh config.yaml

# Cluster (SLURM): submit arrays, then merge
bash run_cluster.sh config.yaml          # submits true + perm arrays
bash run_merge.sh                         # LOSO only — see caveat below

# Cross-pipeline figures
cd ../scripts && python run_all_plots.py
```

**Smoke test before any real run**: `--dry_run` loads/filters data, prints shapes, class balance
and the subject list, then exits — no classifier is ever fit:

```bash
python run_within_subject_classification.py --config config.yaml --dry_run
python run_loso_classification.py --config config.yaml --dry_run
```

**Single-combination override** (useful for debugging one cell of the design):

```bash
python run_loso_classification.py --config config.yaml \
    --contrast ON_vs_OFF_within_median --family all --model_type rf

python run_within_subject_classification.py --config config.yaml \
    --contrast valence_within_median --family all --model_type rf --skip_permutation
```

**⚠ Cluster-mode merge gap (verified, both pipelines have caveats here):**
- **Within-subject**: `run_cluster.sh` references a `merge_ws_results.py` that **does not exist**
  anywhere in the tree. If you split true runs / permutations across a SLURM array, nothing
  aggregates them into the consolidated `*_ws_subject_metrics_averaged.csv` /
  `*_consolidated_sample_predictions.csv` files — only `run_local.sh` (sequential, self-aggregating)
  currently produces trustworthy consolidated output. **Build `merge_ws_results.py`** (port
  `loso_pipeline/merge_loso_results.py`) before relying on within-subject cluster runs.
- **LOSO**: `merge_loso_results.py` *exists* and runs, but its true-run reader looks for
  `{model}_loso_summary.csv` per job while the pipeline actually writes
  `{model}_loso_{n_runs}runs_summary.csv` (the total-run-count is baked into every per-job filename,
  even single-run SLURM-array jobs) — so the merge script's true-run p-value computation currently
  cannot find its inputs. The permutation-side reader does not have this bug. **Always prefer
  `run_local.sh` for numbers you intend to publish** until this filename mismatch is fixed, and
  independently re-verify any `*_pvalues.csv` the merge script produced before trusting it (§13 has
  the full list of citations for this and other bugs found while writing this section).

---

## Conventions (enforced — see root `CLAUDE.md`)

- **No hardcoded paths/params** — everything flows from `config.yaml`; each run writes `used_config.yaml`.
- **No `try/except`** in scientific code — let errors surface.
- **Explicit `random_state`** on every stochastic step.
- **Results isolation** — outputs only under `results/`, never in source dirs.
- **Color palette** — load `color_palette.yaml` (repo root); dimension = hue, significance = fill
  (solid = sig, hollow = n.s.). Never hardcode hex. Confirmed loaded correctly at
  `utils/plotting_utils.py` (`Path(__file__).resolve().parents[2] / "color_palette.yaml"`).
- **Envs** — `ML` for classification, `plots` for figure-only scripts, `eeg`/`junifer` for upstream features.

---

## How to extend the pipeline

1. **New probe dimension / contrast** → add a key under `label_contrasts` + `dimensions`
   (or `run_contrasts`) in *both* configs. `restrict_to` for `_offtask` variants.
2. **New marker family** → add to `feature_families` with `epoch_types` + `prefixes`; add the name to
   `run_families`. Matching is substring-based (see Core concepts above), so pick prefixes that don't
   collide with each other.
3. **New model** → extend `build_model_pipeline` in `utils/ml_utils.py` and the `--model_type` choices.
4. **New diagnostic plot** → add to `utils/plotting_utils.py`, call from the pipeline's plotting block,
   register in `scripts/run_all_plots.py`. Honor the palette/significance convention.
5. **Always** add a test in the relevant `tests/` and run `type1_error/` calibration when changing the
   statistics. Keep the SLURM array math (`array_idx = subject_idx*4 + task_idx`, upstream pipelines
   only — this classification pipeline's own array indexing is per run/perm index, not subject×task)
   in sync with the Python entrypoint.

See each sub-pipeline's README for its run modes and SLURM layout.

---

## Detailed Methods (manuscript-ready description)

This section documents the pipeline at the level of detail expected in a *Nature*-style Methods
section, verified directly against the current code (not against memory or prior documentation).
Where the two CV regimes (within-subject, LOSO) diverge, both are described explicitly. §12–13 flag
exactly what is provisional vs. solid.

### 1. Scientific question and design

Section 2 (CBPT, `Stats_andrillon/`) establishes which MW probe dimensions leave a group-level
neural trace and where. This pipeline asks the complementary question: is that trace strong enough
to decode MW state **in a specific person, trial by trial**? Two independent cross-validation
regimes are run in parallel over the same feature space and the same five probe dimensions:

- **Within-subject** — can a classifier trained and tested *inside one person's own trials* separate
  their MW states above chance? Answers "is the signature usable for this individual."
- **Leave-One-Subject-Out (LOSO)** — can a classifier trained on *N−1* people generalize to a held-out
  person? Answers "is the signature shared across people, or idiosyncratic."

The gap between the two AUCs quantifies the idiosyncratic (non-transferable) component of each
dimension's neural signature (see §12).

### 2. Participants and feature extraction (upstream)

Cohort: subjects `"02"`–`"43"` (42 requested), tasks `Sart1`–`Sart4`. EEG features are produced by
the upstream Junifer pipeline (`junifer_markers/`) as per-probe, pre-probe-window (`-5..-1`)
aggregates and are consumed as-is; this pipeline does not re-derive or re-filter the epoch window.
Per-contrast, per-family CSVs are loaded from `sub-{subject}/eeg/{junifer|junifer_aggregated}/`
(both old- and new-style Junifer output naming are supported).

Two feature representations exist upstream: `per_roi` (features aggregated into anatomical
ROI/electrode-group means — the format actually used by the classification pipeline,
`data_format: per_roi`) and `per_channel` (one column per individual electrode — used only by the
spatial-decoding searchlight variant, §9).

### 3. Marker families

| Family | `epoch_types` | Marker prefixes |
|---|---|---|
| `all` (**the only family actually run**) | sleep, evoked | N1, P1, P3a, P3b, PE(θ/α/β/γ), Kolmogorov complexity, PSD-relative (δ/θ/α/β/γ), wSMI(θ/α/β/γ), slow waves |
| `erp` | evoked | P1, N1, P2, P3a, P3b |
| `sleep` | sleep | slow waves, spindles |
| `spectral` | sleep | band power |
| `connectivity` | sleep | wSMI (θ/α/β/γ) |
| `information_theory` | sleep | permutation entropy, Kolmogorov complexity |

For the `onoff` contrast, `all`/`per_roi` yields **304 candidate features** over **1,216 pre-probe
trials from 29 subjects** (verified by loading the pipeline's own cached feature matrix,
`results/MW_Classification/data_cache/on_vs_off_within_median__all.pkl`); other dimensions have
similar feature counts (~304) with fewer trials/subjects depending on the contrast (see §12, Table
1). Unlike the CBPT analysis (Section 2), which reports signal *per marker family* to build a
family-level omnibus test, the classification pipeline pools **all families together** (`all`) to
answer a single question — "is there enough signal to decode at all" — not "which family carries
it"; per-family decodability (`erp`/`sleep`/`spectral`/…) is configured but not part of the current
reported analysis.

### 4. Label construction — probe-dimension contrasts

Implemented in `create_label_contrast()` (`utils/data_utils.py`). The raw 0–100 MDES score for a
dimension is turned into a binary target by one of four methods, selected per contrast in config:

- **`within_subject_median`** (the method used for every reported contrast): each subject's own
  median for that dimension is computed; a `gap` (raw-scale units; **gap = 5** for every content
  dimension) defines a neutral zone `[median−gap, median+gap]` that is **dropped** entirely (not
  relabeled) — this is a genuine exclusion of ambiguous, near-median trials, not a tie-breaking rule.
  With `gap = 0` the split degenerates to a strict median cut with ties assigned to the positive class.
- **`global_median`** — same idea but the split point is the pooled median across all subjects
  (defined in config but not part of `run_contrasts`).
- **`threshold`** — fixed cut (e.g. at 50) regardless of subject-level distribution (defined, inactive).
- **`extreme_groups`** — a `threshold_low`/`threshold_high` band excludes the middle range
  (defined, inactive).

**Quadratic ("U-shaped") content contrasts** (`valence_sq`, `time_sq` — active in `run_contrasts` for
both pipelines, paralleling the CBPT `valence_sq` marker family in Section 2): the raw score is first
transformed to `(x − 50)² / 50` (a monotone measure of *extremity* from the midpoint), optionally
residualized against the raw linear score via a per-subject OLS fit (`midpoint_sq_residual` —
defined for `valence_sq_res`/`time_sq_res` but **not currently in `run_contrasts`** in either
pipeline), and then the same `within_subject_median` binarization is applied to the transformed
score. These test whether *extremity* of valence/time (not direction) is decodable.

**`restrict_to`** (used for `_offtask` variants, currently commented out / inactive in both configs):
recursively binarizes a helper contrast (typically `onoff`) and restricts the sample to one side of
it (e.g., "off-task probes only") *before* binarizing the real target — i.e., classify e.g. valence
*conditional on* being off-task.

**Per-subject inclusion filters**, applied in this order after binarization: (1) drop subjects with
fewer than `min_samples` (= 10) total probes; (2) drop subjects whose minority-class fraction is
below `min_minority_ratio` (= 0.2), or who have only one class after the gap exclusion. Every
excluded subject and reason is recorded in `subject_exclusions.yaml` / `used_config.yaml`'s
`_data_provenance` block (full transparency, per root `CLAUDE.md`). See §12, Table 1 for realized
cohort sizes per contrast.

### 5. Classification model, preprocessing, and feature selection

`RandomForestClassifier` is the only model actively run (`run_models: ["rf"]` /
`model_type: ["rf"]` in both configs; `xgb`/`lr`/`ocsvm`/`iforest` are implemented and can be swapped
in via config for sensitivity checks). Shared hyperparameters (identical in both pipelines):
`n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features="sqrt",
bootstrap=True, class_weight="balanced_subsample"`.

Every fold refits, inside a single `sklearn`/`imblearn` `Pipeline` (so no step ever sees test-fold
data): **scaler** (`StandardScaler`) → **feature selection** → **(PCA, disabled)** → **(SMOTE, only
when `oversampling_scope="global"` — not the deployed setting, see §6)** → **classifier**.
Feature selection is a two-stage, per-fold procedure: a linear-mixed-model-based univariate prefilter
narrows the candidate pool (controlled by `lmm_prefilter_factor = 3`), then
minimum-Redundancy-Maximum-Relevance (mRMR) selects the final `k = 20` features actually given to the
classifier.

**Missing data** is handled before feature selection, in this order: (1) NaN in `density`-type
markers is zero-filled first (a marker's NaN there means "no event detected," a true zero, not a
missing value); (2) any column with global NaN fraction above `max_feature_nan_frac = 0.25` is
dropped entirely; (3) residual NaNs are imputed per-subject-then-globally by mean (LOSO config) or
median (within-subject config) — a deliberate difference between the two pipelines' configs, not a
bug, though worth stating explicitly if reporting both in the same manuscript. LOSO additionally
z-scores each feature **within participant** before pooling across subjects
(`scale_by_participant: within`) to remove between-subject offsets before the pooled model is fit;
the within-subject pipeline does not need this (`scale_by_participant: none`) since no cross-subject
pooling occurs within a fold.

### 6. Handling class imbalance

- `class_weight="balanced_subsample"` on the Random Forest (both pipelines).
- **SMOTE** (`k_neighbors=5`) is applied with `oversampling_scope="within"` in both configs: rather
  than a single pooled SMOTE call, the training fold is oversampled **independently per subject**
  (`apply_within_subject_oversampling`), balancing each subject's own class distribution; a subject
  with only one class, or too few minority samples to support `k_neighbors`, is skipped for that
  step (with `k` adaptively capped). Synthetic rows inherit their source subject's ID. This is
  applied strictly after the train/test split, only to the training partition.
- **Confidence-based sample weighting** exists in the shared engine
  (`compute_within_subject_confidence_weights`, within-subject min–max normalization of the raw 0–100
  confidence rating) but is **implemented and wired up only for the LOSO pipeline**, gated by a
  per-contrast `confidence_weight.enabled` flag that is **`false` by default and in every deployed
  contrast** — confirmed both in `loso_pipeline/config.yaml` and in an actual run's
  `used_config.yaml`. The within-subject pipeline has no equivalent mechanism at all. Treat any
  confidence-weighted run as exploratory and always report it alongside the unweighted result (see
  memory `loso-pipeline-decisions-2026-06`).
- One-class variants (`ocsvm`, `iforest`) sidestep imbalance by training on a single class only
  (config `oneclass_target`); not part of the currently reported analysis.

### 7. Cross-validation: within-subject vs. LOSO

**LOSO** (`utils/ml_utils.py::run_model_pipeline_cv`): `sklearn.model_selection.LeaveOneGroupOut`
grouped by subject — one fold per subject (`n_folds = n_subjects`, i.e. 24–32 depending on the
contrast's realized cohort, §12 Table 1). Fold membership is **deterministic** — identical across
every one of the `n_runs = 100` seeded passes. Each fold explicitly asserts zero subject overlap
between train and test (`ValueError` otherwise). What actually varies across the 100 seeds is the
stochastic machinery around a fixed fold structure: feature-selection tie-breaking, model
initialization, and per-subject SMOTE draws. Folds within a run execute in parallel
(`joblib.Parallel`, `backend="loky"`).

**Within-subject** (`utils/ml_utils.py::run_within_subject_cv`): for each subject *independently*,
`StratifiedKFold(n_splits=5, shuffle=True, random_state=<run seed>)` over that subject's own
pre-probe trials (alternative `GroupKFold`-by-task and `RepeatedStratifiedKFold`, n_repeats=3, are
implemented but not the deployed strategy). A subject is silently skipped for a given run if their
minority-class count in that run falls below 5 (the fold count) — this is why the number of subjects
contributing scoreable folds in the final results (§12 Table 1) can be slightly lower than the
cohort surviving the §4 inclusion filters (e.g. `valence`: 27 pass inclusion filters upstream, but
only 23 ever have ≥5 minority-class trials across the 100 runs). Unlike LOSO, the **actual K-fold
partition itself changes** across the 100 seeded runs (because `shuffle=True`), so `n_runs` here
captures both fold-formation variability and fitting-procedure variability, not just the latter.

Both regimes report the same per-fold metric set — AUC (`roc_auc_score`, default 0.5 if a test fold
is single-class), AUPRC (`average_precision_score`), MCC (`matthews_corrcoef`), balanced accuracy,
precision/recall/F1 (`zero_division=0`) — unweighted-averaged across folds, then across the 100
seeded runs.

### 8. Statistical inference: permutation testing and multiple-comparisons correction

**True-value distribution.** For each `(contrast, family=all, model=rf)` combination, `n_runs = 100`
seeded passes (fold structure as in §7) yield 100 values of the run-mean AUC (and AUPRC/MCC/balanced
accuracy). This distribution — not a single point estimate — is the "true" statistic; its mean and
spread are reported in §12.

**Null distribution.** `permutation_runs = 500` (LOSO) / `n_permutations = 500` (within-subject) draws.
Labels are shuffled **within subject** in both regimes by default — i.e., each subject's own class
balance is preserved in every null draw (LOSO also supports a fully-global shuffle via
`permutation_scope`, but `within` is the deployed setting in every current config/run). Each draw is
scored with one full CV pass (identical pipeline to a true run, `n_runs=1` for that draw), producing
one null value per draw — 500 per contrast.

**p-value.** Three formulas coexist in the codebase and it matters which one you cite:
1. The internal LOSO group-level statistic (`utils/analysis_utils.py`, and `merge_loso_results.py`)
   uses the add-one, never-zero Phipson & Smyth (2010) estimator:
   `p = (1 + #{perm ≥ true_mean}) / (1 + n_perm)`.
2. The internal within-subject group-level statistic uses a related but distinct, **not** add-one
   formula: `p = #{perm ≥ true_mean}/n_perm` if any exceedance exists, else `1/(n_perm+1)`.
3. **The manuscript figure** (`scripts/generate_combined_classification_figure.py::empirical_pvalue`,
   the source of every number in §12) uses yet a third convention: `p = mean(perm ≥ median(true))` —
   compared against the **median**, not the mean, of the true-run distribution, and with **no
   add-one correction** (so it can be exactly 0). This is the number actually reported below; be
   explicit about this choice in any manuscript text, since it is measurably different from (1)/(2)
   whenever the true-run distribution is skewed or `n_exceed = 0`.

**Multiple-comparisons correction.** Raw p-values from formula (3) are Benjamini–Hochberg FDR
corrected (`statsmodels.stats.multitest.multipletests`, `method="fdr_bh"`) **across the 5 probe
dimensions**, once per pipeline (within-subject and LOSO each form their own 5-test family — never
pooled across pipelines, models, or families, since only `rf`/`all` is active). For the
**individual-level** (per-subject) statistic, the same formula (3) is computed per subject against
that subject's own 500-draw null, and BH-FDR corrected **across subjects within each dimension**
(23–31 tests depending on the dimension) — this is the basis of the "N/M subjects significant"
figures in §12. No correction is currently applied jointly across dimensions and subjects, or across
pipelines.

### 9. Spatial searchlight (per-electrode decoding)

`{loso,within_subject}_pipeline/spatial_decoding/` reuses the identical CV/permutation engine but
restricts the feature matrix to one electrode's own columns at a time (`per_channel` data format),
producing one AUC per channel rather than one AUC over the pooled `all` family. Its own
`config.yaml` differs from the parent pipeline in ways specific to a 64-channel search
(`n_runs=20` for LOSO / within-subject, lighter than the parent's 100; `k=10` selected features
instead of 20).

Significance is a **max-statistic family-wise error rate (FWER)** correction across all channels: for
each of the 500 within-subject-shuffled permutation draws, the *maximum AUC over all channels* is
recorded, building a null of "best possible channel by chance"; the 95th percentile of that null is
the significance threshold, and each channel's own p-value is
`(1 + #{max-null ≥ channel AUC}) / (1 + 500)`. Two subtleties worth stating precisely if reporting
this analysis: (a) the true statistic actually tested against the max-null is each channel's
**single-run** AUC (matching the single-pass nature of each permutation draw), while the AUC
**displayed on the topomap** is the `n_runs`-averaged value — the two are highly correlated but not
identical, and a manuscript figure should say which is which; (b) this spatial multiple-comparisons
correction has no counterpart in the main (`all`-family) pipeline, which reports one pooled AUC per
contrast with no electrode-wise search.

### 10. Type-I error calibration

`{loso,within_subject}_pipeline/type1_error/` empirically calibrates the pipeline's own p-values
under a **known-true null**: real feature covariance is preserved (a multivariate Gaussian is fit
per subject, Ledoit-Wolf shrinkage by default, and synthetic features are drawn from it — this
keeps each subject's covariance structure and sample size intact) while the real labels are retained
unchanged, so by construction there is no feature–label association. The full production pipeline
(CV + permutation testing, unmodified) is then run `n_simulations = 200` times on this synthetic
data (LOSO: `n_runs=3` true passes / `permutation_runs=200` per simulation; within-subject:
`n_runs=3` / `n_permutations=50` per simulation — fewer, for compute budget), and the empirical false
positive rate at `α ∈ {.01, .05, .10}` (with a Wilson 95% CI) is compared to the nominal rate. A
properly calibrated pipeline should show empirical FPR ≈ nominal α at every level.

**Known caveat**: the within-subject Type-I error harness reads `min_minority_ratio` from the wrong
config sub-block (`cv:` instead of the top-level key), which silently resolves to `0.0` — i.e., the
per-subject minority-ratio filter that the production pipeline applies (`0.2`) is **not** applied
during this calibration. This does not invalidate the CV/permutation-machinery calibration itself,
but means the calibration's subject-inclusion behavior does not perfectly mirror production; worth
fixing before citing an exact FPR number from this harness in a manuscript.

### 11. Post-hoc cross-decoding

`cross_decoding/` builds a dimension × dimension generalization matrix **without retraining anything**:
it reuses each dimension's already-saved out-of-fold predictions
(`*_consolidated_sample_predictions.csv`), joins a training dimension's predicted score to a test
dimension's true label on shared probes, and computes the per-subject AUC (averaged across subjects
with ≥8 overlapping probes containing both classes). A dedicated permutation null shuffles scores
within `(subject, train-label class)` groups to preserve the train/test label correlation while
destroying extra ranking information. **This is the critical caveat**: because MDES probe dimensions
are themselves correlated (e.g. off-task trials skew toward certain valence/time responses),
off-diagonal generalization in this matrix is largely explained by that label correlation, not by a
shared neural code — see memory `cross-decoding-is-label-correlation`. Do not present an off-diagonal
cell as evidence of neural specificity without first partialling out label correlation.

### 12. Current results snapshot

**Read this before citing any number below.** These were recomputed on 2026-07-27 by re-implementing
`scripts/generate_combined_classification_figure.py`'s own `load_group_data` /
`load_subject_data` / `empirical_pvalue` / `fdr_correct` functions against the raw per-run and
per-permutation CSVs currently on disk (never the `*_summary_averaged.csv` / top-level
`*_runs_summary.csv` convenience files — both were found to be unreliable, see §13). Both pipeline
configs have uncommitted local changes on this branch, so **these numbers may not reflect the final,
intended analysis** — re-run this recomputation (script referenced in §14) after the configs are
finalized and committed, and reconcile against whatever numbers are ultimately reported in the paper.

**Table 1 — realized cohort per contrast** (after §4 inclusion filters; within-subject "scoreable"
count reflects the additional per-run minority-count-≥5 dropout described in §7):

| Dimension | N requested | N passing inclusion filters (= LOSO cohort) | N scoreable in within-subject runs |
|---|---:|---:|---:|
| On/Off-Task | 42 | 29 | 29 |
| Valence | 42 | 27 | 23 |
| Self/Other | 42 | 32 | 31 |
| Time | 42 | 28 | 27 |
| Confidence | 42 | 24 | 23 |

**Table 2 — group-level AUC, within-subject** (100 true runs / 500 permutations; BH-FDR across the
5 dimensions):

| Dimension | Mean true AUC | Null mean AUC (SD) | p (FDR) |
|---|---:|---:|---:|
| On/Off-Task | 0.710 | 0.500 (0.025) | < .001 |
| Valence | 0.662 | 0.500 (0.028) | < .001 |
| Confidence | 0.652 | 0.499 (0.031) | < .001 |
| Self/Other | 0.596 | 0.498 (0.026) | < .001 |
| Time | 0.582 | 0.499 (0.030) | .006 |

**Table 3 — individual-level significance, within-subject** (each subject's own empirical p vs. their
500-permutation null, BH-FDR across subjects within dimension):

| Dimension | N subjects | N significant (FDR < .05) |
|---|---:|---:|
| On/Off-Task | 29 | 13 |
| Confidence | 23 | 6 |
| Valence | 23 | 5 |
| Time | 27 | 4 |
| Self/Other | 31 | 3 |

**Table 4 — group-level AUC, LOSO** (100 true runs / 500 permutations; BH-FDR across the 5 dimensions):

| Dimension | Mean true AUC | Null mean AUC (SD) | p (FDR) |
|---|---:|---:|---:|
| Confidence | 0.612 | 0.501 (0.034) | < .001 |
| On/Off-Task | 0.589 | 0.503 (0.026) | < .001 |
| Valence | 0.523 | 0.501 (0.032) | .252 |
| Time | 0.522 | 0.499 (0.031) | .252 |
| Self/Other | 0.521 | 0.500 (0.026) | .252 |

**Within-vs-LOSO gap (the "idiosyncrasy" statistic)**: `onoff` 0.710 (within) → 0.589 (LOSO), a gap
of 0.121 AUC. This is qualitatively consistent with the previously-documented narrative ("a MW
signature exists at the group level but carries a large idiosyncratic component"), but the gap is
**larger** than the 0.075 previously reported (0.703→0.628) — re-verify which run is canonical before
restating the smaller figure.

**Substantive changes vs. previously documented numbers, flagged explicitly:**
- Previously: `confidence ≈ selfother` (both ≈0.63). Now: `confidence` (0.652) clearly exceeds
  `selfother` (0.596) at the group level, and the individual-significance ranking also reorders
  (confidence 6/23 > valence 5/23 > time 4/27 > selfother 3/31 — selfother is now the *weakest* of
  the four content dimensions by subject count, not tied for second).
- Previously, `LOSO` results were reported "(onoff only)". The current results directory has LOSO
  runs for **all five dimensions**, and — a genuinely new pattern, not just noise — `confidence`
  generalizes across subjects in LOSO (0.612, FDR-significant) slightly *better* than `onoff` does
  (0.589), while `valence`/`selfother`/`time` do not survive FDR correction in LOSO at all despite
  being clearly above chance within-subject. If confirmed, this reframes the Section 3 narrative:
  it is not simply "onoff decodes best," but "onoff and confidence are the two dimensions whose
  neural signature is *shared across people*, while valence/selfother/time signatures — real at the
  individual level — appear to be subject-specific." **This needs independent confirmation against a
  clean, committed-config run before it goes in a manuscript.**

### 13. Known reproducibility caveats (engineering, verified while writing this section)

These do not necessarily invalidate the numbers in §12 (the manuscript figure script sidesteps most
of them by reading raw per-run files directly), but should be fixed or at least understood before
the pipeline is treated as frozen:

- **LOSO merge script filename mismatch** — `merge_loso_results.py`'s true-run reader looks for
  `{model}_loso_summary.csv`; the actual files are `{model}_loso_{n_runs}runs_summary.csv`. The
  permutation-side reader is unaffected. No `*_pvalues.csv` exists anywhere in `results/`, consistent
  with this path never having been successfully exercised on the current dataset.
- **Within-subject cluster merge step does not exist** — `run_cluster.sh` references
  `merge_ws_results.py`; no such file exists anywhere in the repository (confirmed by `find`). Only
  `run_local.sh` currently self-aggregates correctly.
- **`rf_ws_permutation_500perms_summary_averaged.csv` is not an average** — for every within-subject
  contrast checked, this file contains a single row (`run_idx=499`, the *last* permutation), not the
  aggregate of 500. Anyone reading this file directly gets a noisy single-draw null instead of the
  real one. The manuscript figure script is unaffected (it re-aggregates from the 500 raw
  `permuted_runs/run_*/*_summary.csv` files itself), but this convenience file should either be
  fixed or removed.
- **At least one LOSO top-level `*_runs_summary.csv` is similarly a single leftover run, not a true
  100-run aggregate** — verified for `ON_vs_OFF_within_median`: the file has exactly one row
  (`run_idx=0`), matching that run's own per-run file byte-for-byte, rather than the expected
  100-row (or pre-averaged 100-run) summary. The §12 numbers were computed by re-aggregating the
  100 individual `true_runs/run_*/*_summary.csv` files directly rather than trusting this file, for
  every contrast, in both pipelines — do the same for any future recomputation.
- **`positive_class_name`/`negative_class_name` from `label_contrasts` are defined in
  `loso_pipeline/config.yaml` but never read** — `run_loso_classification.py` always passes the
  hardcoded pair `"ON-task"/"OFF-task"` into the run/permutation functions regardless of contrast.
  This only affects saved metadata labels and confusion-matrix axis text, not the underlying target
  encoding or AUC direction (which come from `positive_above` in `data_utils.py` and are correct) —
  but don't trust the `positive_class_name` column in a saved summary CSV for a non-`onoff` contrast.
- **`xgb_n_jobs: -1` in config is silently overridden to `1`** — `build_model_pipeline` hardcodes
  `n_jobs=1` for `rf`/`xgb`/`lr`/`iforest` regardless of config; fold-level parallelism comes entirely
  from `joblib.Parallel` over CV splits, not from per-estimator multithreading.
  Only relevant if you switch `model_type` to `xgb`.
- **Per-fold label re-binarization is dead code** — a `y_raw`/`use_fold_rebinarize` code path exists
  end-to-end in the within-subject true-run CV function but is hardcoded off (`False`) and never
  actually re-binarizes anything inside the fold; the docstrings describing "per-fold
  re-binarization (no label leakage)" do not match current behavior for true runs (the permutation
  path *does* use it correctly). Labels are fixed once at data-prep time, which is the correct/
  intended behavior — just not what some in-code docstrings claim.
- Several stale comments reference an old `n_runs=10` baseline (both configs currently ship
  `n_runs=100`) — harmless, but worth cleaning up so they don't mislead a future reader.

### 14. Reproducing the §12 snapshot

The recomputation script used for §12 duplicates (does not reimplement independently)
`scripts/generate_combined_classification_figure.py`'s `load_group_data`, `load_subject_data`,
`empirical_pvalue`, and `fdr_correct` functions, reading directly from
`results/MW_Classification/{WithinSubject,LOSO}/<contrast>/all/rf/{true_runs,permuted_runs}/run_*/`.
It intentionally avoids the plotting dependencies (`kaleido`/`plotly`) so it can run in any env with
`pandas`/`numpy`/`statsmodels` (e.g. `ML`). Re-run it (or, once the configs are finalized, the real
figure script via `python scripts/generate_combined_classification_figure.py` in the `plots` env)
any time the underlying `results/` directory changes, and update §12 accordingly — do not hand-edit
the numbers in §12 without re-deriving them the same way.
