# Confidence-based sample weighting (LOSO MW classification)

**Date:** 2026-06-02
**Status:** Implemented (TDD), pending review

## Goal

Use the per-probe **confidence** rating as a `sample_weight` in the LOSO
classifier, so that higher-confidence (more reliable) probe reports weigh more in
the training loss. Confidence is treated as a label-reliability proxy. Rationale
is strongest for the `ON_vs_OFF*` contrasts (confidence pertains chiefly to the
on/off judgement); applying it to content dimensions (valence/time/selfother) is
**exploratory**.

## Design decisions (agreed)

1. **Mechanism:** soft `sample_weight` (not a hard confidence filter).
2. **Normalization:** within-subject min-max → `[w_min, 1]`. Isolates trial-level
   reliability from between-subject baseline confidence. Constant-confidence
   subject → all-ones (neutral). In LOSO each subject is wholly in train or test,
   so within-subject normalization computed once globally equals the per-fold
   value for training subjects — no leakage (test is never weighted).
3. **SMOTE synthetic samples:** weight interpolated. Implemented via 1-NN on the
   subject's original samples (each output row inherits the weight of its nearest
   original; originals map to themselves at distance 0). Does **not** perturb
   SMOTE's feature-space geometry and is order-independent.
4. **class_weight:** kept as-is (`balanced_subsample`). `sample_weight` and
   `class_weight` coexist (orthogonal roles: imbalance vs label noise).
5. **Scope:** per-contrast `confidence_weight` block; can be enabled on any
   contrast. Absent / `enabled: false` reproduces the unweighted pipeline exactly.

## Determinism guards (explicit errors, no silent fallback)

- `oversampling_scope: global` + weighting → error (global SMOTE lives inside the
  `ImbPipeline` and cannot propagate weights). Supported: `within` / `none`.
- one-class models (`ocsvm`, `iforest`) + weighting → error (no labels).
- missing `confidence` column or NaN confidence when enabled → error.

## Config interface

```yaml
ON_vs_OFF_within_median:
  ...
  confidence_weight:
    enabled: true
    normalization: within_subject   # min-max within subject -> [w_min, 1]
    w_min: 0.1
```

## Implementation map

- `utils/ml_utils.py`
  - `compute_within_subject_confidence_weights(confidence, groups, w_min, normalization)`
  - `_interpolate_weights_after_resample(X_original, X_balanced, weights_original)`
  - `apply_within_subject_oversampling(..., weights=, return_weights=)`
  - `_process_cv_fold_loso(..., sample_weights=None)` → `clf__sample_weight`
  - `run_model_pipeline_cv(..., sample_weights=None)` + validation
- `utils/analysis_utils.py`
  - `run_distribution_analysis(..., sample_weights=None)` (forward)
  - `run_permutation_distribution_analysis(..., sample_weights=None)` (null uses
    same pipeline; weights stay attached to trials under label shuffling)
- `loso_pipeline/run_loso_classification.py`
  - `build_confidence_sample_weights(config, contrast_name, df_prepared, groups)`
  - wired into both the true-run and permutation calls

## Tests

`loso_pipeline/tests/test_confidence_weighting.py` (15 tests): weight transform,
SMOTE weight interpolation, LOSO plumbing + determinism errors, config wrapper.

## Scientific reporting requirement

This changes results. Run **with and without** confidence weighting and report
sensitivity (anti-garden-of-forking-paths). Flag content-dimension use as
exploratory.
