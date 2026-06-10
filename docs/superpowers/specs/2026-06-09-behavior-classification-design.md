# Behavior Classification Pipeline — Design Spec

**Date:** 2026-06-09  
**Branch:** feat/spatial-decoding  
**Paper:** Paper 1 — Patients & Behavior

---

## Context

Paper 1 establishes construct validity of the MW probe dimensions using LMMs (Section 1) and group differences (Section behavioral). This pipeline adds a classification layer to ask two complementary questions:

1. **Group decodability:** Can we predict which group a subject belongs to (Controls vs Risk of Depression) from their probe response profile and performance markers?
2. **Intervention decodability:** Can we predict whether a block was an inclusion or exclusion instruction from probe responses + performance markers?

Both use LOSO cross-validation (leave-one-subject-out) so that the test unit is always a subject not seen during training. Results are accompanied by permutation tests (500 perms, +1 convention) to assess significance. Two simple models are compared: Logistic Regression and Random Forest.

This pipeline intentionally mirrors the structure of `mw_classification_pipeline/` but is self-contained in `Behavior/Classification/` — the EEG pipeline is too complex for this behavioral use case.

---

## Architecture

```
Behavior/Classification/
├── config.yaml                    ← all parameters, no magic numbers
├── utils.py                       ← shared LOSO engine, data loaders, result savers
├── run_group_classification.py    ← Task 1: Controls vs Risk of Depression
└── run_ie_classification.py       ← Task 2: Inclusion vs Exclusion
```

Results go to:
```
results/Behavior/classification/
├── group/
│   ├── subject_level/{lr,rf}/
│   └── probe_level/{lr,rf}/
└── ie_intervention/{lr,rf}/
```

Each results folder contains: `summary.csv`, `subject_metrics.csv`, `feature_importances.csv`, `permutation_aucs.csv`, `used_config.yaml`, `plots/`.

---

## Data Sources

| File | Used for |
|------|----------|
| `results/Behavior/probe_data/probe_level_aggregated_data.csv` | Probe dimensions (onoff, valence, selfother, time, confidence) + group + IE label |
| `results/Behavior/probe_data/pca_results.csv` | PC1, PC2, PC3 (off-task probes only, onoff ≤ 50) |
| `results/Behavior/objective_markers/probe_level_with_objective_markers.csv` | omission_rate, commission_rate, rt_mean, rtcv per probe window |

**Key data facts:**
- 42 subjects (Controls=30, Risk of Depression=12) — class imbalance 2.5:1
- 2460 probe-level rows total
- PCs available for 888 off-task probes (subset of probe_level)
- Objective markers file has same 2460 rows, same index

---

## Feature Set

**Core probe features (always included):**
```
onoff, valence, selfother, time, confidence
```

**Objective/performance features (configurable via `include_objective_markers`):**
```
omission_rate, commission_rate, rt_mean, rtcv
```

**PCA features (configurable via `include_pca`):**
```
PC1, PC2, PC3
```
For subject-level or block-level aggregation: mean(PC1), mean(PC2), mean(PC3) across off-task probes for that unit.  
For probe-level: PC1/PC2/PC3 are NaN for on-task probes → impute with 0 (or column mean; controlled by config).

All features are z-scored per training fold (StandardScaler fit on train, transform test).

---

## Task 1 — Group Classification

**File:** `run_group_classification.py`

**Target:** `group` column — binary: `"Risk of Depression"=1`, `"Controls"=0`

**Two modes** (both run by default, controlled by `group_classification.modes` in config):

### Mode A: Subject-level
- Aggregate probe features per subject (mean across all probes)
- For PCs: merge `pca_results.csv`, compute mean(PC1,PC2,PC3) per subject from off-task probes
- For objective markers: aggregate from `probe_level_with_objective_markers.csv`
- **X shape:** (42, n_features) — one row per subject
- **y shape:** (42,) — group label
- **groups:** subject IDs (each subject = 1 data point → 42 LOSO folds of 41 train / 1 test)
- Aggregate fold probabilities across all 42 folds, compute AUC globally

### Mode B: Probe-level
- Each probe is one sample; label = the subject's group
- **X shape:** (2460, n_features)
- **y shape:** (2460,) — group per probe (constant within subject)
- **groups:** subject IDs — LOSO leaves out all probes from one subject
- Aggregate fold probabilities across all folds, compute AUC globally

**Data loading sequence:**
1. Load `probe_level_aggregated_data.csv` (probe dimensions + group)
2. Merge objective markers from `probe_level_with_objective_markers.csv` on `[subject, task, probe_number]`
3. Merge PCs from `pca_results.csv` on `[subject, task, probe_number]` (NaN for on-task probes)
4. Select feature columns per config, impute PCs if needed

---

## Task 2 — I/E Intervention Classification

**File:** `run_ie_classification.py`

**Target:** `inclusion_exclusion` column — binary: `"inclusion"=1`, `"exclusion"=0`

**Data filtering:** Only keep rows where `inclusion_exclusion in ["inclusion", "exclusion"]` (drop baseline)

**Aggregation:** Mean of all features per subject × task block

- **X shape:** (~84, n_features) — one row per subject × intervention block (~42 subjects × 2 blocks)
- **y shape:** (~84,) — inclusion/exclusion label per block
- **groups:** subject IDs — LOSO leaves out both blocks of one subject (~2 test points per fold)

**Data loading sequence:**
1. Load `probe_level_aggregated_data.csv`, filter to `inclusion_exclusion != "baseline"`
2. Merge objective markers, merge PCs (mean per off-task probes within that block)
3. Aggregate to subject × task level
4. Build X, y, groups arrays

---

## LOSO + Permutation Engine (`utils.py`)

### `run_loso(X, y, groups, model, scaler_type="standard")`
```python
logo = LeaveOneGroupOut()
y_true_all, y_proba_all, subject_metrics = [], [], []
for train_idx, test_idx in logo.split(X, y, groups):
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    model.fit(X_train, y[train_idx])
    proba = model.predict_proba(X_test)[:, 1]
    # per-subject metrics (only if enough samples for AUC)
    subject_metrics.append(...)
    y_true_all.extend(y[test_idx])
    y_proba_all.extend(proba)
global_auc = roc_auc_score(y_true_all, y_proba_all)
global_bacc = balanced_accuracy_score(y_true_all, (np.array(y_proba_all) >= 0.5).astype(int))
return {"auc": global_auc, "balanced_accuracy": global_bacc,
        "y_true": y_true_all, "y_proba": y_proba_all,
        "subject_metrics": subject_metrics}
```

### `run_permutations(X, y, groups, model, n_perms, random_state)`
- For each permutation: shuffle `y` globally (preserving group sizes per fold is not required here given small N)
- Re-run full LOSO (including refitting scaler per fold)
- Return array of shape `(n_perms,)` with permutation AUCs

### `compute_pvalue(true_auc, perm_aucs)`
```python
# Phipson & Smyth +1 convention
return (1 + np.sum(perm_aucs >= true_auc)) / (1 + len(perm_aucs))
```

### `build_model(model_type, config)`
```python
if model_type == "lr":
    return LogisticRegression(
        C=config["models"]["lr_C"],
        penalty=config["models"]["lr_penalty"],
        solver="lbfgs", max_iter=config["models"]["lr_max_iter"],
        class_weight="balanced", random_state=config["models"]["random_state"]
    )
elif model_type == "rf":
    return RandomForestClassifier(
        n_estimators=config["models"]["rf_n_estimators"],
        max_depth=config["models"]["rf_max_depth"],
        class_weight="balanced",
        random_state=config["models"]["random_state"]
    )
```

---

## Plots

Each run generates the following in `plots/`:

| Plot | Description |
|------|-------------|
| `metric_distributions.png` | Side-by-side violin/box plots of per-subject AUC and balanced_accuracy — true run vs permutation distribution overlaid |
| `permutation_distribution_auc.png` | Histogram of permutation AUCs + vertical line at true AUC + p-value annotation |
| `permutation_distribution_bacc.png` | Same for balanced accuracy |
| `subject_predictions.png` | Strip/swarm plot: per-subject predicted probability colored by true group/label — one point per subject |
| `feature_importances.png` | Horizontal bar chart (RF: feature_importances_ mean across folds; LR: abs(coef_)) |
| `confusion_matrix.png` | Confusion matrix (counts + normalized) from pooled LOSO predictions |

Plots produced by `utils.py:plot_results(results, output_dir)`.  
`probe_level` mode for group classification is optional (config flag `group_classification.run_probe_level: false` by default).

---

## Config (`config.yaml`)

```yaml
data:
  probe_data_path: "../../results/Behavior/probe_data/probe_level_aggregated_data.csv"
  pca_results_path: "../../results/Behavior/probe_data/pca_results.csv"
  objective_markers_path: "../../results/Behavior/objective_markers/probe_level_with_objective_markers.csv"
  results_root: "../../results/Behavior/classification"

features:
  probe_dimensions: [onoff, valence, selfother, time, confidence]
  include_pca: true
  pca_impute_value: 0.0           # fill NaN PCs for on-task probes (probe_level mode)
  include_objective_markers: true
  objective_marker_cols: [omission_rate, commission_rate, rt_mean, rtcv]

group_classification:
  modes: [subject_level, probe_level]
  target_col: group
  positive_class: "Risk of Depression"

ie_classification:
  target_col: inclusion_exclusion
  positive_class: inclusion
  exclude_conditions: [baseline]
  aggregation_level: [subject, task]   # mean per subject × block

models:
  types: [lr, rf]
  random_state: 42
  lr_C: 1.0
  lr_penalty: l2
  lr_max_iter: 2000
  rf_n_estimators: 100
  rf_max_depth: 4

permutation:
  n_perms: 500
  random_state: 42
  scope_group: global          # shuffle all labels — group is between-subject
  scope_ie: within_subject     # shuffle within subject — IE is within-subject (paired)

loso:
  n_jobs: 4                           # parallel folds via joblib
```

---

## Output Files

Per `{task}/{mode}/{model}/`:

| File | Contents |
|------|----------|
| `summary.csv` | true_auc, true_bacc, perm_auc_mean, perm_auc_std, p_value |
| `subject_metrics.csv` | subject, n_test_samples, auc (if computable), bacc, y_true, y_proba |
| `feature_importances.csv` | feature, importance (mean across folds) |
| `permutation_aucs.csv` | perm_idx, auc |
| `used_config.yaml` | exact config used for reproducibility |
| `plots/` | roc_curve.png, permutation_distribution.png, feature_importances.png, subject_predictions.png |

---

## Verification

1. **Unit test data loading:** assert expected shapes (42 subjects, expected feature count)
2. **Sanity check group classification:** with all features, subject-level AUC should be > 0.5 (groups have known behavioral differences)
3. **Permutation test control:** run with 5 perms locally to confirm p-value computation and file structure before full 500-perm run
4. **I/E classification baseline:** expect near-chance AUC given very small N (~84 block-level points) — result is scientifically valid regardless
5. **Run:** `conda activate ML && python Behavior/Classification/run_group_classification.py --config Behavior/Classification/config.yaml`
6. **Check:** `results/Behavior/classification/group/subject_level/lr/summary.csv` exists and has valid AUC

---

## Notes

- Class imbalance (Controls=30, Risk=12): handled via `class_weight="balanced"` in both models + report balanced_accuracy as primary metric alongside AUC
- Subject-level LOSO (42 folds of 1 test point each): AUC computed by pooling all 42 fold predictions — standard practice for small-N LOSO
- Permutation test shuffles labels globally; within-subject shuffling not needed here (label is not within-subject variable for group classification)
- For I/E task: labels ARE within-subject (each subject has both inclusion and exclusion) → permutation should shuffle within-subject to preserve the paired structure; controlled by `permutation.scope: within_subject` in config
- No `try/except` blocks — let errors surface
- All stochastic operations use `random_state` from config
