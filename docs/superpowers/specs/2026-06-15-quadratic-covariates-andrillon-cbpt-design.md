# Quadratic (extremity) covariates for bipolar dimensions in the Andrillon CBPT pipeline

**Date:** 2026-06-15
**Scope:** `Stats_andrillon/` (Paper 2, Section 2 — EEG CBPT). Behavior (Paper 1) is an out-of-scope follow-up, documented at the end.
**Status:** Design approved; pending spec review → implementation plan.

## 1. Goal

`valence` and `time` are **bipolar** probe dimensions on a 0–100 scale (0 = negative/past, 100 = positive/future, 50 = neutral/present). The current CBPT model tests only their **linear** (monotonic, polarity) effect. We want to additionally test a **quadratic** effect — the squared distance from the midpoint — which captures **extremity/intensity** independent of polarity:

- `valence_sq` → emotional intensity (strong feeling regardless of sign)
- `time_sq` → temporal distance (far past *or* far future vs. present)

Both should be added as **fixed covariates in a single full model** and each tested as its own predictor of interest, producing its own CBPT topographic map.

This change folds into the pending 5000-permutation rerun (one pass; MCC = FDR-BH per family, already wired).

## 2. Locked decisions

1. **Single full model.** The squared terms are fixed covariates in the default formula for *every* predictor's test (not per-predictor formulas). Existing onoff/valence/selfother/time maps are recomputed under the augmented model.
2. **Only `valence_sq` and `time_sq`** (selfother excluded by choice).
3. **Definition:** `x_sq = (x − 50)² / 50` (midpoint 50; `/50` rescales to 0–50). Same definition already used by `Behavior/Objective_Markers/lmm_probe_dimensions.py::compute_time_squared`.
4. **Global orthogonalization** of each squared term against its linear partner (one pooled residualization over the analysis rows). Within-subject orthogonalization is **rejected** — it is not a global linear reparametrization, so it breaks the highest-order-term invariance and distorts the variable per subject.
5. **Variability filter** for a squared predictor is applied on its **raw source dimension** (`valence` for `valence_sq`, `time` for `time_sq`) at the existing threshold (`min_predictor_variability = 30`), so the linear and quadratic tests of a dimension run on the **same subjects** and remain comparable.
6. **6 CBPT maps** per run: `onoff, valence, valence_sq, selfother, time, time_sq`.

## 3. Scientific rationale

### 3.1 Why orthogonalize, and what it does
In `y ~ x + x²`, raw `x` and `x²=(x−50)²/50` are collinear when the data are not symmetric around 50. Measured on the real data:

| dimension | corr(raw, sq) global | corr within-subject (median) | VIF(raw, sq) |
|---|---|---|---|
| valence | +0.825 | +0.863 | 3.13 |
| time | +0.699 | +0.441 | — |

(valence mean = 65.7, time mean = 59.6 → both shifted off the 50 midpoint, hence collinearity.)

Global orthogonalization replaces `x_sq` with its residual after a **single pooled** linear fit on `x`:
```
x_sq_orth = x_sq − (â + b̂·x)        # â, b̂ from one OLS over all analysis rows
```
`x_sq_orth` is uncorrelated with `x` by construction (global corr → 0, VIF → 1).

### 3.2 Highest-order-term invariance (the key guarantee)
The t-statistic of the **highest-order term** in a polynomial regression is invariant to a global linear reparametrization of the lower-order terms. Verified empirically (OLS, random y): the quadratic t is identical for raw vs. globally-orthogonalized (`−0.751` in both); only the linear t changes.

**Consequence:** the **extremity/curvature map (`valence_sq`/`time_sq`) is identical whether or not we orthogonalize.** Orthogonalization is purely a cleanup of the **linear (polarity) map** — it removes the collinearity-inflated SE and restores power and a clean "total polarity" interpretation, making the linear valence/time maps comparable to the onoff/selfother linear maps (which have no quadratic partner).

Within-subject orthogonalization breaks this (quadratic t shifted `−0.751 → −0.072`), confirming it is the wrong choice.

### 3.3 Interpretation of the 6 maps
| predictor | meaning |
|---|---|
| `onoff` | attention (off ↔ on task) — unchanged |
| `valence` | polarity (negative ↔ positive) |
| `valence_sq` | emotional extremity/intensity (orthogonal to polarity) |
| `selfother` | self ↔ other — unchanged |
| `time` | temporal direction (past ↔ future) |
| `time_sq` | temporal distance (near ↔ far from present) |

## 4. Architecture

Approach **A** (config-driven, isolated to the Andrillon pipeline; the shared `Statistics/reader.py` gets one small backward-compatible parameter).

### 4.1 Config (`Stats_andrillon/config_andrillon.yaml`)
New section, mirroring the existing `derived_ratios` pattern:
```yaml
derived_covariates:
  valence_sq:
    source: valence
    midpoint: 50
    scale: 50
    orthogonalize: global   # residualize against `source` over the analysis rows
  time_sq:
    source: time
    midpoint: 50
    scale: 50
    orthogonalize: global
```
Default formula updated to:
```
power ~ onoff + valence + valence_sq + selfother + time + time_sq + time_on_task + confidence + (1|subject)
```
`predictor_of_interest` list extended with `valence_sq` and `time_sq`.

### 4.2 New pipeline helpers (`Stats_andrillon/andrillon_pipeline.py`)
- `_get_derived_covariates_lookup(config) -> Dict[str, spec]` — parse/validate the config section (each spec needs `source`, `midpoint`, `scale`, `orthogonalize`).
- `_add_raw_squared_columns(df, lookup)` — add raw `name = (df[source] − midpoint)**2 / scale` for every derived covariate. Called on the loaded behavioral frame **before** `prepare_data_for_lmm`, so the columns exist for (a) the formula and (b) the predictor-of-interest existence check.
- `_orthogonalize_squared_columns(df_behavioral, lookup) -> (df, vif_report)` — for each derived covariate with `orthogonalize: global`, replace the column in `df_behavioral` with the residual of a pooled OLS `name ~ source` computed **on the exact rows returned by `prepare_data_for_lmm`** (per-marker, so orthogonality holds on the design matrix actually fitted). Also compute and return the VIF of `source` and the orthogonalized term.

### 4.3 Order of operations in `run_marker_analysis`
1. `load_all_probe_data` → `df_filtered`.
2. `_add_raw_squared_columns(df_filtered, lookup)` (raw squared columns).
3. `prepare_data_for_lmm(..., predictor_of_interest=POI, variability_column=<source if POI is derived else POI>)` → `power_data, df_behavioral, channels`.
4. `_orthogonalize_squared_columns(df_behavioral, lookup)` → orthogonalized columns + VIF report. (Runs for **all** markers, since both squared terms are always in the formula.)
5. Per-subject normalization, LMM fit, permutations, clustering — unchanged.

The **derived-ratio marker path** (`_load_ratio_marker_data`) gets the same steps 2 and 4 on its returned `df_behavioral` (valence/time columns are present there).

### 4.4 Shared reader change (`Statistics/reader.py`)
Add `variability_column: Optional[str] = None` to `prepare_data_for_lmm`; when `None` it falls back to `predictor_of_interest` (current behavior — backward compatible). The variability filter (`filter_subjects_by_variability`) uses `variability_column`. The Andrillon pipeline passes the raw `source` for derived predictors.

### 4.5 Formula validation
`_resolve_formula_for_predictor` already substring-checks that the predictor is in the formula; `valence_sq`/`time_sq` are present in the default formula, so no change needed. (Note: `"valence" in formula` still passes for the `valence` predictor even with `valence_sq` present — substring check is harmless here.)

### 4.6 VIF reporting
Persist the per-marker VIF report (source vs. orthogonalized term) into `results['diagnostics']` and write a `*_collinearity.csv` next to the other per-marker CSVs, so the near-1 VIF after orthogonalization is auditable. Expected: VIF ≈ 1 for the orthogonalized terms.

## 5. Outputs & compute

- New model folders (formula string drives `get_model_folder_name`), one per predictor → 6 folders per run, including `..._target_valence_sq` and `..._target_time_sq`.
- MCC: unchanged — FDR-BH per family via `apply_mcc_postprocessing`, persisted into `results.pkl`, consumed by `generate_summary_report`.
- Compute: +2 predictors × all markers × 5000 permutations. With the corrected SLURM allocation (16 cpus, 48 h), well within budget.

## 6. Testing plan

1. **Unit (synthetic):** `_add_raw_squared_columns` produces `(x−50)²/50`; `_orthogonalize_squared_columns` yields corr(source, orth) ≈ 0 and VIF ≈ 1; idempotent re-run.
2. **Invariance check:** on real behavioral data, confirm the quadratic-term t-stat from an OLS is unchanged between raw and globally-orthogonalized squared column (already verified manually; encode as a test).
3. **Variability routing:** assert that for predictor `valence_sq`, the subjects retained equal those retained for predictor `valence` (filter on raw source).
4. **Pipeline smoke test:** one marker, small `n_permutations`, `n_jobs=1`, confirm the 6-term formula fits, VIF CSV written, clusters produced.
5. **No regression:** running with the squared terms removed from config reproduces the previous formula/behavior.

## 7. Risks / open points

- **Residual within-subject collinearity** after global orthogonalization (valence within-subject corr ≈ −0.46 → VIF ≈ 1.26): mild, accepted, and reported via the VIF CSV. Global is the standard, defensible choice; within-subject is rejected (Section 3.2).
- **Interpretation discipline (anti-HARKing):** the squared terms are pre-registered here as planned covariates + tests, not post-hoc. Report all 6 maps regardless of significance.
- **Subject count for squared predictors:** filtering on the raw source keeps them aligned with the linear tests; if a dimension's raw filter is already lenient, no extra subjects are lost.

## 8. Out of scope — Behavior follow-up (recommended)

For cross-paper consistency, the same recipe should later be applied in `Behavior/Objective_Markers/lmm_probe_dimensions.py`: add `valence_sq`, switch the existing `time_sq` sensitivity term (and the new `valence_sq`) to global orthogonalization. **The existing `time_sq` quadratic result is not invalidated** (highest-order invariance — the quadratic t is identical); only the linear term in that sensitivity model and the addition of `valence_sq` are new. Tracked as a separate task (Paper 1), not blocking the EEG rerun.
