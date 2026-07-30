# Cross-dimension residualized classification contrasts

Date: 2026-07-30
Status: approved

## Motivation

Section 3 (individual-level decodability) classifies each MDES dimension
(onoff, valence, selfother, time, confidence) independently via median split.
The dimensions are correlated with each other at the probe level (e.g. onoff
and valence). This raises the concern that when the classifier decodes
valence, part of that signal may actually be attributable to onoff (or
another correlated dimension) rather than to valence-specific neural
information.

To test this, we add a residualized variant of each of the 5 base
dimensions: before the usual within-subject median split, the dimension is
OLS-residualized (per subject) against the other 4 dimensions. This isolates
the variance in that dimension that is *not* explained by the others. This
is additive: it does not modify or replace the existing (non-residualized)
contrasts or their results.

This is a natural extension of an existing pattern already in
`utils/data_utils.py` (`midpoint_sq_residual`), which residualizes
`valence_sq`/`time_sq` against their own linear dimension to isolate
curvature. That transform is untouched — it serves a different purpose
(isolating a quadratic term from its own linear component) and is not
being generalized into this one.

## Scope

**In scope:**
- All 5 base linear dimensions get a residualized variant:
  `onoff_within_median_res`, `valence_within_median_res`,
  `selfother_within_median_res`, `time_within_median_res`,
  `confidence_within_median_res`.
- Each is residualized against the other 4 (e.g. `valence` is residualized
  against onoff, selfother, time, confidence).
- Applies to both `within_subject_pipeline/config.yaml` and
  `loso_pipeline/config.yaml` (both read `create_label_contrast` from the
  same shared `utils/data_utils.py`).

**Out of scope:**
- The quadratic contrasts (`valence_sq`, `time_sq`, `valence_sq_res`,
  `time_sq_res`) are not touched and do not get a cross-dimension variant.
- Plotting/figure scripts (`generate_combined_classification_figure.py`,
  `make_fig_section3_decoding.py`) are not updated to visualize the new
  contrasts in this pass — data generation only.

## Design

### Transform: `linear_residual`

New branch in `create_label_contrast()` (`utils/data_utils.py`), parallel to
the existing `midpoint_sq` / `midpoint_sq_residual` branches.

Contrast config gains two new keys:
- `transform: linear_residual`
- `residualize_against: [<col1>, <col2>, <col3>, <col4>]` — explicit list of
  the other dimension column names. Explicit in config rather than inferred
  inside `data_utils.py`, per the project's no-hardcoded-parameters rule —
  the function stays generic and the config is the single source of truth
  for which dimensions correlate with which.

Per subject:
1. Take rows where the target dimension and all columns in
   `residualize_against` are non-NaN.
2. If the count of such rows is `< min_valid_for_residual` (new config key,
   default 15 — matches the existing `min_valid.sum() < 3` guard pattern
   used by `midpoint_sq_residual`, scaled up: 5 model parameters here
   (4 slopes + intercept) vs. 2 there, so the threshold scales by the same
   ~1.5x samples-per-parameter ratio, giving ~10 residual degrees of
   freedom): **exclude that subject from this contrast** (residual = NaN for
   all their rows). Per explicit user decision: subjects without enough
   probes to residualize reliably should not enter classification for this
   contrast, rather than fall back to unresidualized raw values (which
   would silently mix two different constructs within the same contrast).
3. Otherwise, fit OLS (`np.linalg.lstsq`, same approach as the existing
   transform, robust to the collinearity among the 4 predictors) and replace
   the target column with residuals for that subject's valid rows.
4. Individual rows with a missing predictor (in an otherwise-included
   subject) get NaN and are dropped downstream by the existing missing-data
   handling — same as today.

After residualization, the existing `split_method` (`within_subject_median`,
`gap`, etc.) runs unmodified on the residualized column.

### Config changes

In both `within_subject_pipeline/config.yaml` and
`loso_pipeline/config.yaml`:
- Add `min_valid_for_residual: 15` near the other experiment-params.
- Add 5 new entries to `label_contrasts`, mirroring their non-residualized
  twins' `split_method`/`gap`/`positive_above`/class names, e.g.:

  ```yaml
  valence_within_median_res:
    column_name: valence          # label_source: valence for LOSO
    transform: linear_residual
    residualize_against: [onoff, selfother, time, confidence]
    split_method: within_subject_median
    gap: 5
    positive_above: true
    positive_class_name: "positive"
    negative_class_name: "negative"
  ```

- Add the 5 new contrast names to `run_contrasts`, appended after the
  existing entries (existing entries and their order are untouched).

### Results location

No new pipeline, no new results root. Results land in the same tree as
every other contrast: `results/MW_Classification/WithinSubject/<contrast>/`
and `results/MW_Classification/LOSO/<contrast>/`, keyed by the new contrast
names.

### Interpretation note (for later reporting, not implemented here)

A `_res` contrast answers "does EEG carry information about dimension X that
is *not* explained by the other 4 dimensions?" — a different, more
conservative question than the raw contrast. Expect lower AUC than the
non-residualized twin. Must not be silently merged into the main
decodability-hierarchy figure without this caveat.

### Smoke test (this pass)

A throwaway local config (not committed, or committed under a clearly
marked `_smoketest` name) limited to the 5 new `_res` contrasts, with
`n_runs` reduced and permutations disabled (`outputs.run_permutations:
false` / `permutation_runs: 0`), run locally to confirm the transform
executes end-to-end and produces plausible AUCs before considering a full
(100-run, permutation-backed) run.

## Testing

Unit tests for the `linear_residual` transform (mirroring the existing
`TestCreateLabelContrastThreshold` style in
`loso_pipeline/tests/test_data_utils.py`):
- Residualizing a dimension perfectly correlated with a predictor collapses
  residuals to ~0 (up to numerical precision) after removing the mean.
- A dimension independent of its predictors is left ~unchanged by
  residualization.
- A subject below `min_valid_for_residual` is excluded (residual all-NaN for
  that subject) rather than falling back to raw values.
- Rows with a missing predictor value are NaN in the output regardless of
  the subject's overall validity count.
- Downstream `within_subject_median` split still runs correctly on the
  residualized column (gap/positive_above behavior unchanged).
