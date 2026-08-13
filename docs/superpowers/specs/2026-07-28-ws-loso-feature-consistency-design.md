# WS↔LOSO feature consistency — design

**Date**: 2026-07-28
**Status**: design approved; blocked on the WS re-run (SLURM 3185788/3185789/3185790)
**Scope**: `mw_classification_pipeline` — replace the broken WS-vs-LOSO feature-importance
comparison with an analysis that can actually detect partial consistency.

---

## 1. Why the current analysis is replaced, not patched

The three combined figures (`results/combined_figures/scatter_feature_importance.png`,
`scatter_shap_directional.png`, `scatter_shap_absolute.png`) show what looks like a strong
*negative* relationship between within-subject (WS) and LOSO feature importance. It is an
artifact. Four independent problems were confirmed numerically before this redesign.

### 1.1 The plotted `r` is computed on a selection-on-extremes subsample

`generate_combined_classification_figure.py` annotates Pearson `r` over
`_select_top_union` = (top-10 by WS) ∪ (top-10 by LOSO), not over the common feature set.
For `onoff`: `r_full = 0.039` (n=177) vs `r_top10union = -0.859` (n=20).

A permutation test (1000 draws, shuffling the WS↔LOSO feature pairing) confirms this is pure
selection geometry, not a residual bug:

| quantity | value |
|---|---|
| null `r_top10union` | mean **-0.793**, SD 0.090, range [-0.945, -0.296] |
| observed -0.859 | **24.5th percentile** of that null (p = 0.245) |
| zero overlap between top-10 sets under the null | **56.1%** of draws — the modal outcome |
| null draws with zero overlap | `r` = -0.844 ± 0.050 |

The same pattern holds across all five linear dimensions (`r_full` weakly **positive** 0.04–0.23,
`r_top10union` strongly negative -0.42 to -0.86), which is what a generic mechanical artifact
looks like.

**The join itself is correct** — `_common_feature_indices` matches by feature name, never
positionally. Directional SHAP sign conventions are also consistent (`positive_above: true` in
both configs). Those hypotheses were checked and ruled out.

### 1.2 Zero-inflation asymmetry

Both pipelines zero-pad features that mRMR did not select in a fold. Measured on `onoff`:

```
WS   run_0: 304 features,   1 exact zero (0.3%),  top-10 = 9.8% of total importance
LOSO run_0: 177 features, 132 exact zeros (74.6%), top-10 = 51.7%
```

Both pipelines select `k = 20` features per fold. LOSO's 29 folds converge on nearly the same
ones (45 distinct non-zero across the whole run), so the other 132 stay at exactly 0. WS's 29
subjects each select a *different* 20, so the union covers nearly every column and the mean is
almost flat. The "LOSO concentrates signal, WS is diffuse" shape is
arithmetic, not a finding. Correlating a 75%-zero vector against a near-uniform one is near-zero
by construction.

### 1.3 Averaging across subjects destroys the quantity of interest

If each subject relies on an idiosyncratic subset — which the WS-vs-LOSO AUC gap already implies —
the cross-subject mean represents no actual model. Comparing that smear against a single
group-level profile is a category error.

### 1.4 Gini importance dilutes across collinear features

EEG markers are heavily correlated (adjacent ROIs, adjacent bands, wSMI channel pairs,
`psd_relative_*` summing to 1). RF splits arbitrarily within a correlated cluster, so two models
can use the *same underlying signal* and land on different columns. At 177-column granularity
agreement is underestimated by construction.

**Conclusion**: `r_full ≈ 0.04–0.23` is a floor imposed by the method, not an estimate of
consistency. Patching the plot would keep three of the four problems.

---

## 2. Prerequisite (in flight)

WS results on disk were computed on a stale 304-column feature space; LOSO used the declared 177.
Fixed 2026-07-28 and all 7 WS contrasts relaunched. See
`memory/classification-canonical-feature-space-177.md`. **This analysis must not run until those
jobs finish** — otherwise mRMR picked ~45 features from a pool of 304 on one side and 177 on the
other, depressing overlap for a purely mechanical reason.

---

## 3. Design

### 3.1 Unit of analysis: the subject, paired

For each subject *S* of 29, over the 177 common features:

| side | source | meaning |
|---|---|---|
| `WS(S)` | `WithinSubject/{contrast}/all/rf/true_runs/run_*/rf_loso_{S}_shap_values.pkl` → `mean(\|SHAP\|)` over S's trials, averaged over 100 runs | how **S's own model** uses the features |
| `LOSO(S)` | `LOSO/{contrast}/all/rf/true_runs/run_*/rf_loso_100runs_shap_values.pkl`, segmented back to S's fold via `rf_loso_100runs_sample_predictions.csv`, same trials | how the **group model trained without S** uses them |

The stacked LOSO SHAP matrix is `(1216, 177)`. Segment it with
**`rf_loso_100runs_sample_predictions.csv`** — 1216 rows carrying `subject`, `fold_idx` and
`sample_idx`, matching the SHAP rows one-to-one. (Do **not** use
`rf_loso_100runs_fold_predictions.csv`: it holds one row per fold, 29 rows, and cannot map trials
to subjects.) The implementation must assert this row-count match rather than assume it.

Same metric, same trials, same scale. Two properties follow:

- Both sides are built from the same `mrmr k: 20` selection budget per fold (identical in
  `within_subject_pipeline/config.yaml` and `loso_pipeline/config.yaml:320`), so both profiles
  carry comparable, bounded sparsity — **the zero-inflation asymmetry (§1.2) disappears**, because
  it came from averaging 29 heterogeneous selections on one side only. The exact non-zero count
  per side is the union over that side's folds and runs and must be reported, not assumed equal.
- No cross-subject averaging — **§1.3 disappears**.
- LOSO trains without S, so the group view of S is genuinely independent of S. No circularity.

### 3.2 Separate two questions the old analysis conflated

- **Selection overlap** — `J(S)` = Jaccard between the non-zero feature sets.
  Null: **hypergeometric** (random selection of the same sizes from 177).
- **Magnitude agreement** — `rho(S)` = Spearman of `mean(|SHAP|)` over the union of non-zero
  features. Null: permutation of the name-pairing (validated in §1.1).

Reported per subject and aggregated: mean + CI over the 29, plus **count of individually
significant subjects**, mirroring the "16/29" framing already used for AUC.

### 3.3 Noise ceiling — mandatory, never report a bare correlation

Split the 100 runs odd/even, build each profile from each half, correlate half-A vs half-B →
`rel_WS(S)`, `rel_LOSO(S)`, Spearman-Brown corrected to full length. Then:

```
rho_corrected(S) = rho(S) / sqrt( rel_WS(S) · rel_LOSO(S) )
```

Always report **observed, ceiling, and % of ceiling** together. Without this there is no way to
know whether `rho = 0.20` is poor or near-perfect. Also compute the ceiling for the legacy
group-level contrast, so the old `r = 0.04` can be stated as a fraction of its maximum.

*Declared limitation*: this ceiling captures estimation noise from seeds/CV only, not
subject-sampling variability.

### 3.4 Three levels of granularity — pre-specified, not chosen post hoc

The 177 columns decompose exactly into the 23-marker Andrillon CBPT set:

| level | units | purpose |
|---|---|---|
| feature | 177 | maximum granularity (diluted by collinearity, §1.4) |
| **marker** | **23** | the CBPT marker set — collapses collinear ROI clusters |
| family | 2 (evoked=4, sleep=19) | same families as the MCC |

Aggregation to marker level **sums** over ROIs (importance is additive). Null: permute the
feature→marker assignment, preserving marginals.

*Declared caveat*: ERPs have 1–2 ROIs and sleep markers 9, so summing favours sleep markers. For
the **consistency** question the bias is identical on both sides and cancels; it must be declared
if markers are ever ranked against each other.

Because this hierarchy comes from the existing CBPT analysis rather than from inspecting these
results, it is not a forking path.

---

## 4. Deliverables

- `mw_classification_pipeline/scripts/feature_consistency_analysis.py` — config-driven, type
  hints, numpy docstrings, no `try/except`, explicit `random_state` (project CLAUDE.md rules).
- `mw_classification_pipeline/scripts/config_feature_consistency.yaml`.
- Outputs to `mw_classification_pipeline/results/feature_consistency/` (+ `used_config.yaml`):
  - `per_subject_consistency.csv` — subject, J, rho, rho_corrected, rel_WS, rel_LOSO, p_J, p_rho
  - `marker_level_consistency.csv`
  - `noise_ceiling_summary.csv`
- Figures (palette from `color_palette.yaml`; significance by fill, never by hue):
  - distribution of the 29 per-subject `rho` with the ceiling band overlaid
  - marker-level heatmap (23 markers × dimensions)
- Replace the three broken scatters. **No figure may compute a correlation on a top-N subsample.**

## 4b. Implemented 2026-07-30 — deviations from the design above

Recorded here rather than by editing §1–4, so the reasoning that was approved stays legible
next to what the data forced.

| design said | implemented | why |
|---|---|---|
| 177 features | **175** | the 2026-07-28 ERP ROI fix (montage is CACS-64, not BC-32) moved P1/N1 from 2 ROIs to 1. 4 evoked×1 + 19 sleep×9 = 175, verified against the feature names at load time. |
| WS side from `rf_loso_{S}_shap_values.pkl` | `*_shap_values_stacked.pkl` + `*_sample_predictions.csv`, segmented by subject | the per-subject file is named `rf_ws_{S}_…`, not `rf_loso_{S}_…`. The stacked route was verified to reproduce the per-subject pickles exactly and reads ~29× fewer files. Trial sets, counts and labels were confirmed identical between pipelines, keyed on (subject, task, probe_number). |
| Jaccard of non-zero sets + hypergeometric null (§3.2) | **dropped** | both configs now set `feature_selection.method: "none"`, so the forest is fitted on all 175 columns: LOSO gives every feature non-zero SHAP for every trial, the Jaccard is identically 1 and the LOSO selection-frequency vector is constant (rank correlation undefined). Reporting J = 1 would be reporting the config back as a result. Only the per-side count of used features survives, descriptively. §1.2's zero-inflation asymmetry is likewise gone — disabling selection is what removed it. |
| marker-level null = permute the feature→marker assignment | permute the WS↔LOSO **pairing** at marker level | the direct analogue of the feature-level test, so the two levels answer the same question at different resolutions. |
| marker-level heatmap (23 × dimensions) | scatter grid of marker shares, WS vs LOSO | the designed heatmap cell (across-subject ρ per marker) is **non-significant for every marker in every dimension** after FDR — a 23×7 grid of blanks. The scatter shows the group-level marker agreement, which is where the result actually is. |

### Result

Group-level agreement is strong and rises sharply with marker aggregation:

| dimension | n | mean per-subject ρ | sig subjects | group ρ (175) | group ρ (23) | ceiling |
|---|---|---|---|---|---|---|
| On/Off-Task | 29 | 0.083 | 6/29 | 0.413 | 0.762 | 0.995 |
| Valence² | 14 | 0.062 | 2/14 | 0.391 | 0.873 | 0.993 |
| Time² | 13 | 0.067 | 2/13 | 0.390 | 0.651 | 0.992 |
| Valence | 23 | 0.076 | 4/23 | 0.348 | 0.753 | 0.988 |
| Confidence | 23 | 0.012 | 2/23 | 0.307 | 0.780 | 0.972 |
| Self/Other | 31 | 0.019 | 1/31 | 0.220 | 0.659 | 0.992 |
| Time | 27 | −0.015 | 0/27 | 0.003 (n.s.) | 0.427 | 0.991 |

So §1.4 was the dominant problem: most of the feature-level disagreement is arbitrary choice
among collinear ROI columns of the same marker. The per-subject correlations stay near zero
regardless, which is the genuine idiosyncrasy result.

### Found while implementing

`utils/ml_utils.py` only populated RF Gini importances inside its `feature_selection` branch.
With selection disabled that branch never fires, so **every `*_feature_importances.csv` on
disk is uniformly zero**, and `scatter_feature_importance.png` was correlating two zero
vectors (surfacing only as an opaque `SVD did not converge` from `np.polyfit`). Fixed via
`_extract_fold_importances`, covered by `loso_pipeline/tests/test_fold_importances.py`;
the on-disk CSVs need a pipeline re-run to become meaningful.

## 5. Out of scope

- Cross-referencing consistency against the CBPT cluster results marker-by-marker. Dropped: the
  marker set is already shared, and the extra step was not judged worth its cost.
- Re-running LOSO — it already uses the correct 177.
- Reconciling the Section 3 headline AUCs. Tracked separately in
  `memory/mw-classification-headline-drift-2026-07.md`; they will move because of the feature-space
  fix and must be recomputed with `scripts/recompute_headline_numbers.py`.
