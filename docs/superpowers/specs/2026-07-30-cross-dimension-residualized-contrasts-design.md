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
- `onoff`, `valence`, `selfother`, `time` are each residualized against the
  other **3 content dimensions only** (excluding `confidence` — see
  "confidence is not a predictor" below). `confidence` itself is
  residualized against all 4 content dimensions (onoff, valence, selfother,
  time), since that direction has no such concern.
- Applies to both `within_subject_pipeline/config.yaml` and
  `loso_pipeline/config.yaml` (both read `create_label_contrast` from the
  same shared `utils/data_utils.py`).

### Confidence is not a predictor for the other 4 (added 2026-07-30, post-smoke-test)

The original design residualized every dimension against the other 4,
treating `confidence` as a peer content dimension like valence/selfother/
time. It is not: `confidence` ("confidence in self-assessment") is a
metacognitive judgment about the reliability of the probe report itself,
not an independent axis of experiential content. There is a standing
hypothesis in this project (already reflected in the commented-out
`confidence_weight` mechanism in both configs) that confidence is
*downstream of* onoff — mind-wandering plausibly reduces meta-awareness/
introspective accuracy, so low confidence may be a consequence or
common-effect marker of being off-task rather than an independent cause.

If that causal direction holds, using `confidence` as a predictor when
residualizing `onoff` is adjusting for a mediator/downstream effect of the
very thing being measured — it removes genuine onoff-related signal, not a
nuisance confound (classic over-adjustment / conditioning on a mediator).
This is a real risk specific to `confidence`; it does not apply to valence/
selfother/time, which are independent content dimensions and legitimate
peers of onoff.

Consequence: `residualize_against` for onoff/valence/selfother/time now
lists only the other 3 content dimensions (3 predictors + intercept = 4
params). `confidence_within_median_res` is unaffected — regressing
confidence on onoff/valence/selfother/time has no reverse-causality
problem in that direction (confidence as outcome, the others as candidate
causes), so it keeps all 4 as predictors.

`min_valid_for_residual` is adjusted to match the smaller parameter count
for the 3-predictor contrasts: 12 (~3x the 4 model params, keeping the same
samples-per-parameter ratio used throughout this design), vs. 15 for the
unchanged 4-predictor `confidence_within_median_res`.

The initial smoke test (2026-07-30, WS, 3 runs, all 5 contrasts with
confidence included as a 4th predictor for onoff/valence/selfother/time)
is superseded for those 4 contrasts and was re-run after this change;
`confidence_within_median_res` numbers from the first pass remain valid
since its predictor set did not change.

**Smoke-test evidence for the mediator concern** (WS, 3 runs, no perms,
excluded_subjects=0 throughout both passes — mean AUC across the 3 runs):

| contrast | with confidence as predictor | confidence excluded |
|---|---|---|
| onoff_within_median_res | 0.575 | 0.617 |
| valence_within_median_res | 0.576 | 0.594 |
| selfother_within_median_res | 0.533 | 0.551 |
| time_within_median_res | 0.505 | 0.524 |
| confidence_within_median_res | 0.593 | 0.593 (unchanged, as expected) |

Every contrast's AUC rose once confidence was dropped as a predictor —
`onoff` by the largest margin. This is consistent with (not proof of) the
mediator hypothesis: confidence was absorbing genuine onoff-related signal,
most strongly for onoff itself. `confidence_within_median_res` is
unaffected, as expected, since its predictor set did not change. These are
3-run, no-permutation point estimates — directionally informative, not a
significance claim.

### Production run 1 (2026-07-30): onoff/valence/selfother/time _res

Full production settings (100 runs, 500 permutations) submitted for the 4
linear residualized contrasts in both WS and LOSO. `confidence_within_median_res`
deliberately excluded from this run — see rationale below (reversed after
user follow-up; see "Production run 2"). All 4 job arrays (WS true/perm,
LOSO true/perm — 4800 jobs total) completed with zero failures.

### Extending to the quadratic terms + confidence's own result (2026-07-30, follow-up)

Two follow-up decisions after production run 1 landed:

1. **`valence_sq`/`time_sq` get a cross-dimension residualized variant
   too** (`valence_sq_res_cross`, `time_sq_res_cross`), "completing the
   set" alongside the 4 linear contrasts. New transform
   `midpoint_sq_residual_cross` (in `utils/data_utils.py`, alongside
   `residualize_within_subject`): computes `(x-50)²/50` then residualizes
   it, in ONE joint within-subject OLS, against **both** its own linear
   dimension (added automatically, mirroring `midpoint_sq_residual`) **and**
   the other 3 content dimensions listed in `residualize_against`
   (confidence excluded, same mediator reasoning as above). A single joint
   regression was chosen over two sequential residualization passes (own
   linear, then cross-dimension) to avoid an order-dependent result — one
   regression with all predictors is the standard way to condition on
   several covariates at once. Same subject-exclusion convention as
   `linear_residual` (below `min_valid_for_residual` ⇒ excluded, not
   fallen back to the raw score). `min_valid_for_residual: 15` (4
   predictors + intercept = 5 params, matching `confidence_within_median_res`).
   `valence_sq`/`time_sq`'s own existing residualized variant
   (`valence_sq_res`/`time_sq_res`, own-linear-only) is untouched.

2. **`confidence_within_median_res` is run after all**, reversing the
   "don't run it" call from earlier in this doc. Rationale from the user:
   confidence is still excluded as a *predictor* for the other 4 (mediator
   concern stands), but decoding confidence net of the other 4 is a
   legitimate question in its own right — the earlier exclusion conflated
   "don't use as a predictor" with "don't compute as a target," which
   don't have to be the same decision.

Both changes are additive: `valence_sq`/`time_sq`/`valence_sq_res`/
`time_sq_res` and the 4 linear `_res` contrasts already run are untouched.

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
