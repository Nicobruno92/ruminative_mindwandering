# Spatial Decoding Extension — Design

**Date:** 2026-06-08
**Status:** Draft for review
**Author:** nbruno (with Claude)
**Pipelines affected:** `mw_classification_pipeline/loso_pipeline`, `mw_classification_pipeline/within_subject_pipeline`

---

## 1. Motivation

Section 2 of Paper 2 (CBPT) characterizes mind-wandering through **topographies**: the
main figure is topomaps of marker families × dimensions. Section 3 (classification)
currently reports a single AUC per dimension using spatially-aggregated ROI features
(`data_format: per_roi`). This creates a conceptual and visual gap between the two
sections.

**Spatial decoding** closes that gap. By training one classifier per electrode (a
searchlight over scalp locations) and mapping its decodability, Section 3 produces
topomaps directly comparable to the CBPT maps. The narrative becomes:

> CBPT identifies which topographic patterns characterize each dimension at the group
> level. Spatial decoding asks whether that same pattern is robust enough to predict
> mental state trial-by-trial — and *where* on the scalp it is decodable.

The decodability hierarchy (onoff > valence > ...) gains a spatial interpretation, and
both methods validate each other.

## 2. Goal

Add a **spatial decoding** mode to both the LOSO and Within-Subject (WS) pipelines that:

- Trains one model **per electrode** (64 channels, standard 10-20 montage).
- Each per-electrode model uses **only that channel's markers**, with the existing
  feature pipeline applied within the channel (family filtering + mRMR `k` + SMOTE + CV).
- Runs for **all 5 dimensions**: `onoff`, `valence`, `selfother`, `time`, `confidence`.
- Runs **permutation tests per electrode** for spatial statistical inference.
- Saves per-channel metrics in a topomap-ready format and renders topographies.

This is an *extension* (a subfolder per pipeline), not a modification of the existing
ROI-level pipelines. The existing results remain untouched.

## 3. Non-Goals (YAGNI)

- No new ML engine. The per-electrode model is the *existing* engine restricted to one
  channel's columns. No reimplementation of CV / permutation / feature selection.
- No new classifiers or contrasts beyond what `config.yaml` already defines.
- Spatial **cluster-permutation** correction (à la CBPT) is deferred — FDR is the
  default; cluster correction is a documented future extension.
- No source localization. This is sensor-space scalp decoding only.

## 4. Chosen Architecture (Option C — shared core + thin drivers)

A single shared "searchlight" core in `utils/`, with one thin driver per pipeline that
passes its own analysis function as a callback. This factors the channel-loop,
aggregation, FDR, and topomap logic into one place (DRY) while reusing each pipeline's
existing ML engine unchanged.

```
utils/spatial_decoding_utils.py            # SHARED CORE
    parse_channels_from_columns(X)         # "{channel}_{marker}" -> ordered channel list
    select_channel_columns(X, channel)     # restrict X to one channel's markers
    run_spatial_searchlight(...)           # loop channels, call analysis_fn, aggregate
    fdr_correct(p_values)                  # Benjamini-Hochberg across channels
    build_info_from_channels(channels)     # MNE Info + standard_1020 montage
    plot_channel_topomap(...)              # AUC / significance topomap (one dimension)
    plot_combined_topomap_panel(...)       # 5-dimension panel for the paper

loso_pipeline/spatial_decoding/
    run_loso_spatial_decoding.py           # driver: passes run_distribution_analysis +
                                           #          run_permutation_distribution_analysis
    config.yaml                            # inherits LOSO config, forces per_channel
    run_spatial_slurm.sh
    merge_spatial_results.py               # aggregate per-(dim,channel) jobs -> topomaps

within_subject_pipeline/spatial_decoding/
    run_within_spatial_decoding.py         # driver: passes run_within_subject_* fns
    config.yaml
    run_spatial_slurm.sh
    merge_spatial_results.py

tests/test_spatial_decoding.py             # unit tests for the shared core
```

### Why a callback?

`run_spatial_searchlight` is agnostic to LOSO vs WS. It receives:
- the prepared data (`df`, `X`, `y`, `groups`, ...),
- an `analysis_fn` (the existing distribution-analysis function),
- a `perm_fn` (the existing permutation function),
- a `metric_extractor` callback that pulls the scalar AUC (and per-subject AUC for WS)
  out of whatever the analysis function returns.

For each channel it calls `analysis_fn` / `perm_fn` with `X` restricted to that
channel's columns and `save_plots=save_pickle=save_shap=False`, then records the
metrics. This guarantees per-electrode models are identical to the main pipeline.

## 5. Data Flow (per dimension)

1. Load data **once** with `data_format: per_channel` → columns `{channel}_{marker}`.
2. `parse_channels_from_columns` → ordered list of the 64 channels.
3. For each channel `ch`:
   - `X_ch = select_channel_columns(X, ch)` (all of `ch`'s markers across the active
     family).
   - Call the existing analysis engine on `X_ch` (mRMR `k`, SMOTE, CV, n_runs all from
     config) → capture `mean_auc`, `std_auc`.
   - Call the existing permutation engine on `X_ch` → null distribution + `perm_p`.
   - (WS only) capture per-subject AUC vector and count of individually-significant
     subjects.
4. Assemble `per_channel_metrics` and apply `fdr_correct` across the 64 channels.
5. Save tables + render topomaps.

## 6. Multiple-Comparisons Correction — max-statistic permutation (FWER)

**Decision (revised after an implementation smoke test):** the spatial correction is a
**max-statistic permutation test** controlling the family-wise error rate (FWER) across
the 64 electrodes — the same logic as the CBPT cluster-permutation in Section 2.

**Why not per-channel FDR (the original draft):** two problems surfaced during the
LOSO smoke test.

1. *Statistical power.* A per-channel permutation p has floor `1/(n_perm+1)`. Under
   Benjamini-Hochberg across 64 channels the most-significant electrode's adjusted p is
   `p_raw × 64`. Reaching FDR significance needs `p_raw ≤ 0.05/64 ≈ 0.0008`, i.e.
   `n_perm ≳ 1300` *per channel*. With the main pipeline's 500 permutations the best
   achievable adjusted p is `0.002 × 64 = 0.128` — no electrode could ever be
   significant. The design was underpowered by construction.
2. *Compute.* Measured ~300 s per permutation per channel (LOSO, 29 subjects, k=10,
   contended node). 500 perms × 64 channels × 5 dimensions of independent per-channel
   permutations is prohibitive.

**Max-statistic test.** For each dimension:

- **True statistic:** per-channel AUC averaged over `n_runs` (the searchlight already
  built and validated) → 64 true AUCs.
- **Null of the max:** for each of `N` label permutations, apply *one* within-subject
  label shuffle to **all 64 channels at once**, score each channel with `n_runs=1`, and
  record `max_c AUC_c` for that shuffle → an `N`-length null distribution of the maximum.
- **Inference:** electrode `c` is significant iff `true_AUC_c` exceeds the
  `(1-alpha)` quantile of the max-null. Per-channel FWER p-value:
  `p_c = (1 + #{max_null ≥ true_AUC_c}) / (1 + N)` (the project's +1 convention).

This controls FWER across the scalp with only `N` permutations *total* (not per channel),
is correctly powered, and is directly comparable to the Section-2 CBPT correction.

**Matched estimator (test vs display).** Each permutation scores a channel with a single
LOSO pass (`n_runs=1`). To keep the test calibrated, the true statistic entered into the
FWER comparison is also a *single* pass (`auc_single`, the first run / `run_idx=0`),
matched to the null. The topomap is coloured by the more stable `n_runs`-averaged
`mean_auc` (better visual estimate). Both columns are stored in
`per_channel_metrics.csv`; the `sig` flag derives from `auc_single`, the colour from
`mean_auc`. (Using the averaged AUC against a single-pass null would still control FWER
but lose power — the matched single-pass statistic recovers it.)

**Critical correctness requirement:** within a single permutation, the *same* shuffled
labels must be scored across all 64 channels (so the max is taken over channels for a
common null draw). This is why the permutation parallelization axis is
**(dimension × permutation index)** — one job draws one shuffle and scores all channels —
rather than (dimension × channel). It does not rely on fragile cross-job seed alignment.

`per_channel_metrics.csv` stores `mean_auc`, the FWER `perm_p`, the `sig` flag at the
configured `alpha`, and the global FWER threshold is recorded alongside. Both the raw
per-channel AUC map and the significance-masked map are saved (forking-paths
transparency). Benjamini-Hochberg FDR remains available in the code as a secondary,
optional correction but is not the default.

## 7. Outputs

```
results/MW_Classification/SpatialDecoding/
  {LOSO|WithinSubject}/{contrast}/{family}/{model}/
    per_channel_metrics.csv      # channel, mean_auc, std_auc, perm_p, perm_p_fdr, sig,
                                 #   n_sig_subjects (WS only)
    per_channel_subject_auc.csv  # (WS only) channel × subject AUC matrix
    permutation_nulls.csv        # per-channel null AUC distributions (for re-analysis)
    topomap_auc.png              # AUC per electrode, standard_1020 montage
    topomap_sig.png              # AUC masked by FDR significance
    used_config.yaml
  combined/
    topomap_panel_auc.png        # 5-dimension panel (paper figure)
```

Per-channel artifacts (plots, pickles, SHAP) are **suppressed** to avoid generating
320 heavy subfolders; only scalar metrics and null distributions are persisted.

## 8. SLURM Parallelization

Two arrays per pipeline, reflecting the max-statistic design:

- **True-run array — axis (dimension × channel)** = 5 × 64 = 320 tasks. Each computes
  the `n_runs`-averaged true AUC for one electrode (`--contrast C --channel CH --true`).
  Light and fast.
- **Permutation array — axis (dimension × permutation index)** = 5 × `N` tasks (e.g.
  5 × 500 = 2500, or batched: `B` perms/task to cut the job count). Each task draws ONE
  within-subject label shuffle and scores all 64 channels with `n_runs=1`
  (`--contrast C --perm_idx P`), writing that shuffle's per-channel AUCs (and their max).

A per-channel data cache (`spatial_cache_path`, keyed by contrast/family/data_format)
is precomputed once per contrast so the array tasks load in ~1 s instead of re-reading
and pivoting the CSVs (the dominant cost). `merge_spatial_results.py` then reads the 64
true AUCs and the `N` max-null draws, computes per-channel FWER p-values + the global
threshold, writes `per_channel_metrics.csv`, and renders the topomaps.

## 9. Configuration

Each `spatial_decoding/config.yaml` inherits its parent pipeline's settings and only
overrides:

```yaml
data_format: "per_channel"        # REQUIRED — searchlight needs channel-level columns
spatial_decoding:
  channels: "all"                  # or an explicit list to subset electrodes
  montage: "standard_1020"
  multiple_comparisons:
    method: "fdr_bh"               # FDR Benjamini-Hochberg across channels per dimension
    alpha: 0.05
  feature_selection_k: 10          # mRMR k for per-channel models (default 10, configurable)
  topomap:
    cmap: "RdBu_r"
    vmin_vmax: "auto"              # or [0.4, 0.7] to fix the AUC color scale
```

Feature selection, CV, SMOTE, n_runs, n_permutations, classifiers — all inherited
unchanged from the parent config, **except mRMR `k` which defaults to `10`** for the
per-channel models (smaller per-channel feature pool) and remains configurable via
`spatial_decoding.feature_selection_k` in the subfolder config.

## 10. Testing

`tests/test_spatial_decoding.py`:
- `parse_channels_from_columns` recovers the correct ordered channel set from synthetic
  `{ch}_{marker}` column names (incl. channels whose name is a substring of another,
  e.g. `P1` vs `P10` — must use word-boundary logic).
- `select_channel_columns` returns exactly the target channel's columns and nothing else.
- `fdr_correct` matches `statsmodels` / known BH reference values.
- `build_info_from_channels` builds a valid MNE Info with all 64 channels positioned on
  `standard_1020`.
- A fast end-to-end smoke test on a tiny synthetic dataset (2 channels, ~40 samples,
  2 runs, 5 perms) that exercises `run_spatial_searchlight` with a stub `analysis_fn`.

## 11. Open Decisions (sign-off requested)

1. **FDR across channels per dimension** as the default correction (vs. cluster-spatial).
2. **SLURM axis (dimension × channel) = 320 tasks.**
3. **Suppress per-channel plots/pickle/SHAP** (save only scalar metrics + nulls).
4. mRMR **`k=10`** default for per-channel models, configurable via
   `spatial_decoding.feature_selection_k`. **(Resolved.)**

## 12. Scientific Guardrails Checklist

- No hardcoded paths (use `utils/bids_compliance.py` / config).
- No hardcoded parameters (everything via `config.yaml`).
- No `try/except` in scientific code.
- Explicit `random_state` everywhere (inherited from parent config).
- Both FDR-corrected and uncorrected maps saved (forking-paths transparency).
- `used_config.yaml` written to every results folder.
- Per-electrode models provably identical to the main pipeline (same engine, restricted X).
