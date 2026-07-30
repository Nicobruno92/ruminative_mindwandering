# Pipeline Review Report

**Date**: 2026-07-29
**Pipeline**: `Stats_andrillon/` (code) + `results/andrillon_cluster/` (results)
**Scientific Question**: For each of 7 MW probe dimensions (onoff, valence, valence_sq, selfother, time, time_sq, confidence), tested one at a time as `predictor_of_interest` in `power ~ onoff + valence + valence_sq + selfother + time + time_sq + time_on_task + confidence + (1|subject)`, does the dimension show a group-level neural correlate across 23 EEG markers (4 evoked ERP + 19 sleep/pre-probe spectral+connectivity+info-theory+slow-wave markers), corrected for multiple comparisons (a) marker-wise via BH within each family and (b) at the family level via a permutation-based omnibus test (count + min-p), itself BH-corrected across dimensions? Concretely: is `onoff`'s "17/19 sleep markers significant" result a correctly-computed, trustworthy finding, and is the pipeline ready to report it in Paper 2?
**Expected Direction**: A three-tier hierarchy already documented in `CLAUDE.md`: `onoff` and `valence_sq` "concentrated and localizable" (marker-wise BH survivors + omnibus count/min-p both survive cross-dimension BH ≈0.001, peaking in PE_beta); `time`, `valence`, `time_sq` "distributed but real" (omnibus count near-significant uncorrected, ≈0.04, but BH≈0.052, no single marker FWER-significant); `selfother` weak/null (omnibus p≈0.085).
**Prior Reviews**: None for this pipeline. A review dated 2026-06-05 exists (`reports/report_pipeline_review_2026-06-05.md`) but is for a different pipeline (`mw_classification_pipeline/loso_pipeline/`, the LOSO classifier) and is not directly applicable here.

---

## Executive Summary

- **The BH arithmetic itself is correct, at every level checked.** Manual re-derivation of marker-wise BH on the 19 raw sleep p-values for `onoff` exactly reproduces `multiple_comparisons_summary.csv` (17/19 rejected). The omnibus permutation test's dependence-robust count statistic independently corroborates the same 17/19 result (observed 17 vs. null mean 1.90, p=0.0002) using a completely different assumption set (no independence/PRDS assumption at all). The cross-dimension BH step is also numerically correct on manual re-derivation and matches the three-tier hierarchy in `CLAUDE.md` almost exactly (sleep BH: onoff/valence_sq ≈0.0012, valence/time/time_sq tied at ≈0.0516, selfother ≈0.0852). **So the user's core worry — "did I mess up the correction?" — has a clean answer: no, for the sleep family and for onoff specifically, the correction is sound and is corroborated two independent ways.**

- **But "correctly computed" and "ready to report" are not the same claim, and the gap is concrete and dated.** The evoked family (P1/N1/P3a/P3b) was silently re-fit on 2026-07-28/29 for **all six already-processed targets**, changing cluster composition and p-values materially (e.g. `onoff`/P3b: cluster stat 82.9→105.3, p 0.0038→0.0016; `valence_sq`/P3a: a 2-electrode non-significant cluster (p=0.157) became a 13-electrode cluster at raw p=0.015) — yet `apply_mcc_postprocessing.py`, `omnibus_test.py`, and the per-target `SUMMARY_TOPOPLOTS_*.pdf` were **not** re-run afterward. Every on-disk evoked-family number for all six targets is stale relative to the current raw fits. The sleep family (19 markers, the user's 17/19 headline) was **not** touched by this re-fit and remains internally consistent — that specific number is safe — but the pipeline's own evoked-family conclusions ("weak/null for every non-onoff dimension") are no longer verified.

- **`confidence` (added as a 7th target on this branch) has completed all 23 raw marker fits but has never had the MCC or omnibus step run** — `multiple_comparisons_summary.csv`/`mcc_family_composition.csv` are absent from its results directory, and `omnibus_test.csv` (last generated 2026-07-27) predates it. This is an unfinished pipeline step, not a failure, but it means the sixth-to-seventh-dimension multiplicity correction the user needs before publishing is not yet computable.

- **The Paper 2 main figure generator (`plot_cbpt_summary_figure.py`) is non-functional against the current results tree and methodologically wrong regardless**: its hardcoded model-folder names don't match any directory on disk (missing `valence_sq`/`time_sq`, and two of its six columns point at interaction-model directories that were never run), and — independent of that — it derives "significant" purely from the raw uncorrected cluster p-value (`p < 0.05`), never reading the BH-corrected p-values at all. The on-disk figure is dated 2026-06-09, seven weeks stale.

- **Single most important next step**: re-run `Statistics/apply_mcc_postprocessing.py` and `Stats_andrillon/omnibus_test.py` (no `--model-dir`, so it reprocesses all targets at once) for the six already-"final" targets to pick up the evoked re-fit, **before** quoting any evoked-family number (including "onoff: 19/23") in the paper — the `valence_sq`/P3a change in particular could alter the Section 2 narrative if it survives correction.

**Overall Scientific Quality**: Good (methodology is rigorous and unusually well-documented; the concrete gaps are result-freshness/completeness, not design flaws)
**Key Risk**: Evoked-family conclusions for all six processed dimensions — including the "onoff: 2 evoked markers survive" and "no dimension besides onoff shows an evoked signal" claims that feed the Section 2 narrative — are currently unverified against the pipeline's own most recent raw fits, and at least one cell (`valence_sq`/P3a) moved from clearly non-significant to plausibly significant.

---

## 1. Pipeline Overview

### Stage 1: Configuration and marker resolution
- **Script**: `Stats_andrillon/config_andrillon.yaml`, `Stats_andrillon/andrillon_pipeline.py::get_marker_list`
- **Input**: `feature_families` (named fragments, e.g. `erp: [P1, N1, P3a, P3b]`) × `selected_markers` (which families run per epoch type: `evoked → [erp]`; `sleep → [spectral_relative, connectivity, information_theory, sleep]`). Resolved against markers actually present on disk (`get_available_markers`).
- **What it does**: substring-matches configured fragments against on-disk marker names, in order, first match wins; also injects virtual `derived_ratios` markers.
- **Output**: an ordered list of 23 `"<epoch_type>/<marker>"` strings (4 evoked, 19 sleep).
- **Key decisions**: family membership is declared, not discovered from disk — a marker present on disk but not declared is silently excluded; a marker declared but absent from disk is silently skipped with a warning at this stage (hard error later at MCC time via `require_complete_family`).
- **Scientific assumptions**: the declared family composition (23 = 4+19) is what "the family" means for every downstream correction step.

### Stage 2: Data loading and preprocessing
- **Script**: `andrillon_pipeline.py::load_marker_matrix` (calls `Statistics/reader.py::load_all_probe_data`, `prepare_data_for_lmm`, `Statistics/helpers.py::add_quadratic_features`, `apply_response_transform`, `normalize_by_subject`)
- **What it does, in order**: (1) load raw marker + behavioral data, (2) compute orthogonalized quadratic features `valence_sq`/`time_sq` = `(x-50)²/50` residualized on the linear term via OLS, (3) filter to the requested `marker_type`, (4) `prepare_data_for_lmm` (pivot to wide format, compute `time_on_task`, apply `onoff_max_value` filter, apply the per-subject predictor-variability filter — the only exclusion step that actually removes subjects, N varying 31–37/42 by target), (5) apply the configured response transform (currently `enabled: false`) **before** the per-subject z-score, (6) per-subject z-score normalization.
- **Key decisions**: variability threshold 30 (linear, 0–100) / 15 (quadratic, 0–50 scale); response-transform selection fixed by residual excess kurtosis, independent of any predictor's significance.
- **Scientific assumptions**: per-subject z-scoring is required so residual variance isn't dominated by between-subject amplitude; quadratic term must be orthogonalized to avoid collinearity.

### Stage 3: LMM fit + Freedman-Lane permutation + cluster detection
- **Script**: `andrillon_pipeline.py::run_marker_analysis` (→ `Statistics/lmm_model.py`, `Statistics/cluster_test.py`)
- **What it does**: fits `statsmodels.MixedLM` (or `lme4::lmer` via `rpy2` for WLS, unused) per channel for observed data; fits reduced model once; generates 5000 Freedman-Lane permutations (residuals of reduced model permuted within-subject, full model refit with original predictor labels) with seeds offset to avoid RNG collision with the observed fit; forms candidate clusters via p<0.025 spatial adjacency (`stat_fun="sum"`, `separate_signs=True`, edge channels excluded from cluster formation only); Monte-Carlo p-value with Phipson & Smyth `+1` correction.
- **Key decisions**: `permutation_method: freedman_lane`, chosen because formula covariates are correlated with several predictors of interest.
- **Scientific assumptions**: Freedman-Lane requires within-subject residual exchangeability — the heteroscedasticity (Breusch-Pagan) diagnostic is the check that speaks to this directly (see SCI-005).

### Stage 4: Marker-wise multiple-comparisons correction
- **Script**: `Statistics/apply_mcc_postprocessing.py`
- **What it does**: represents each marker by its max-statistic cluster p-value (unit="marker"; no-cluster marker → p=1.0); BH within each family (evoked m=4, sleep m=19 — `state` folded into `sleep`, see SCI-004); aborts (does not shrink the family) if a declared marker is missing; persists corrected p-values back into pickles; writes `cluster_summary_corrected.csv`, `multiple_comparisons_summary.csv`, `mcc_family_composition.csv`.
- **Key decisions**: BH, not BY/Bonferroni, despite a documented PRDS caveat (SCI-004).

### Stage 5: Family-level omnibus permutation test
- **Script**: `Stats_andrillon/omnibus_test.py`
- **What it does**: `count` (markers with permutation p≤α, null from the same joint permutation draws across markers — no independence assumption) and `min_p`/Tippett (FWER-valid under arbitrary dependence); leave-one-out ranking; when run over all model dirs, also applies BH **across dimensions**, separately per family/statistic.
- **Output**: `results/andrillon_cluster/omnibus_test.csv` — a single file, **overwritten wholesale** on every invocation, not appended (SCI-007).

### Stage 6: Summary report + Paper-2 figure
- **Scripts**: `Statistics/generate_summary_report.py` (per target); `Stats_andrillon/plot_cbpt_summary_figure.py` (cross-target).
- **Key decision/bug**: `plot_cbpt_summary_figure.py` reads the raw, uncorrected `*_clusters.csv` directly and thresholds "significant" at raw permutation p<0.05 — it never reads the BH-corrected p-value (SCI-002).

### Stage 7 (optional): LMM assumption diagnostics
- **Script**: `Stats_andrillon/lmm_assumption_diagnostics.py` — refits the observed model only (cheap), reports descriptive residual-shape diagnostics (skew, excess kurtosis, % outliers, Breusch-Pagan p), deliberately not Shapiro-Wilk.

### Orchestration (SLURM)
`submit_andrillon_predictor_loop.sh` → `submit_parallel_andrillon_markers.sh --predictor <p>` (regenerates `run_andrillon_marker_array.sh`/`run_andrillon_report_generation.sh` in place, submits array + dependent report job) → `run_andrillon_marker_array.sh` (propagates Python exit code so failures register as `FAILED`). This chain is internally consistent but was **not** what generated the current `confidence`/re-run-evoked results — those used ad-hoc, custom-named, untracked scripts instead (SCI-009), most likely due to a documented cluster-wide SLURM `MaxJobCount` congestion the same week.

---

## 2. Results Found

### `multiple_comparisons_summary.csv` × 6 targets
- **Finding** (matches README §8 and CLAUDE.md exactly): `onoff` (N=35/42): **19/23** significant — evoked P3a (p_corr=0.045), P3b (p_corr=0.015); sleep 17/19 (all except PE_theta [0.100] and slowwaves_Slope [0.203]). `valence_sq` (N=31/42): **7/23**, all sleep. `valence`(N=33), `selfother`(N=37), `time`(N=35), `time_sq`(N=34): **0/23** each.
- **Assessment**: **Needs verification** — sleep-family rows are current and trustworthy; evoked-family rows in all six files are stale relative to the current raw fits on disk (SCI-001).

### `omnibus_test.csv` (regenerated 2026-07-27)
- **Finding**: sleep family — onoff count_p=0.0002→BH 0.0012, valence_sq 0.0004→BH 0.0012; valence/time/time_sq ≈0.038–0.043→BH 0.0516 (tied); selfother 0.0852→BH 0.0852. Evoked family — all non-significant or borderline (onoff 0.094→0.198; time_sq 0.033→0.198).
- **Assessment**: **Valid for sleep; needs verification for evoked** (predates the 2026-07-28/29 evoked re-fit). Does not include `confidence`.

### `assumption_summary.csv`
- **Finding**: exists **only for `onoff`** (generated 2026-07-22). `psd_relative_gamma` (exkurt=20.5, 100% heteroscedastic) and `slowwaves_Density` (exkurt=7.3, 100% heteroscedastic) stand out sharply; several other sleep markers also 70–90% heteroscedastic. Convergence ≥0.97 everywhere checked.
- **Assessment**: **Incomplete** — never run for the other 5 targets or `confidence`.

### `target_confidence/` — raw fits complete, no MCC/omnibus
- **Finding**: all 23 marker subfolders present with complete raw outputs (most recent 2026-07-29 04:13, per custom-named SLURM logs `cbpt_conf_*`). Absent: `multiple_comparisons_summary.csv`, `mcc_family_composition.csv`, any `SUMMARY_*` file. `omnibus_test.csv` predates it.
- **Assessment**: **Incomplete, not invalid.**

### `results/figures/cbpt_summary_figure.{png,pdf}`
- **Finding**: dated 2026-06-09 — predates valence_sq/time_sq, the omnibus test, and most current results.
- **Assessment**: **Problematic** (SCI-002).

---

## 3. Scientific Issues

### [SCI-001] Evoked-family raw fits regenerated for all 6 processed targets without re-running correction/omnibus — Severity: CRITICAL
- **Type**: Inconsistency / stale results
- **Location**: `results/andrillon_cluster/*__target_{onoff,valence,valence_sq,selfother,time,time_sq}/evoked_{N1,P1,P3a,P3b}/results.pkl` vs. `multiple_comparisons_summary.csv`, `omnibus_test.csv`
- **What**: All 24 evoked `results.pkl` files (4 markers × 6 targets) have mtimes 2026-07-28 22:53–2026-07-29 02:46; all six `multiple_comparisons_summary.csv` predate this (2026-07-24/25); `omnibus_test.csv` predates it too (2026-07-27). Direct comparison: `onoff`/P3b cluster grew 24→27 electrodes (stat 82.88→105.27, p 0.0038→0.0016); `onoff`/P3a 16→21 electrodes (stat 45.49→62.24, p 0.0224→0.0084 — notable since its current BH-corrected p, 0.045, barely survives); `valence_sq`/P3a went from a non-significant 2-electrode cluster (p_corr=0.157) to a 13-electrode cluster (raw p=0.015) — a qualitative shift. The sleep family (checked for onoff, 19 markers) was untouched, all mtimes 2026-07-24 16:46, matching its MCC summary. The only uncommitted config change is adding `confidence` to `predictor_of_interest`, explicitly documented as leaving "the six existing runs... untouched" — so the root cause of the evoked re-fit could not be established from the repository alone (see SCI-008).
- **Why it matters**: Every evoked-family headline number for six targets — including "evoked weak/null except onoff" and onoff's own "2 evoked markers" — is not reproducible from current data; `valence_sq`/P3a's shift could add a new evoked finding to the paper's narrative.
- **Evidence**: file mtimes; direct content comparison of `cluster_summary_corrected.csv` vs freshly-written `*_clusters.csv` for 3 marker-target cells.
- **Scientific proposal**: Re-run `apply_mcc_postprocessing.py` and `omnibus_test.py` (all targets) before quoting any evoked number; specifically check whether `valence_sq`/P3a survives correction.
- **Confidence**: CONFIRMED (staleness); LIKELY (not confirmed) on exact root cause.

### [SCI-002] Paper 2 main figure generator mismatched to current results and uses uncorrected p-values — Severity: CRITICAL
- **Type**: Bug / Interpretation error
- **Location**: `Stats_andrillon/plot_cbpt_summary_figure.py:84-97` (`DIMS`), `:188-206` (`load_clusters`)
- **What**: `DIMS` hardcodes folder names (e.g. `onoff_valence_selfother_time_time_on_task_confidence__target_onoff`) missing `valence_sq`/`time_sq` — none match any actual directory; two entries reference interaction-model directories (`onoff_x_confidence...`, `valence_x_confidence...`) that don't exist anywhere on disk (confirmed by search). `load_clusters()` silently returns `[]` for missing paths (no error) — a re-run today would produce a blank figure. Independently: significance is decided by raw permutation `p_value < 0.05` from the uncorrected `*_clusters.csv`; the BH-corrected p-value is never read.
- **Why it matters**: This is the literal main figure of Paper 2 Section 2; as written it cannot regenerate correctly and, even path-fixed, would encode the wrong statistical criterion.
- **Evidence**: directory search confirms absence of interaction-model dirs; figure mtime 2026-06-09.
- **Scientific proposal**: Derive `DIMS` programmatically from `get_model_folder_name()`/the actual predictor list (as SLURM scripts already do); switch significance to the BH-corrected marker p-value with the existing hierarchical gate.
- **Confidence**: CONFIRMED

### [SCI-003] `confidence` target: correction/omnibus not yet run — Severity: HIGH
- **Type**: Missing analysis
- **Location**: `results/andrillon_cluster/*__target_confidence/` (missing summary files); `omnibus_test.csv` (excludes confidence)
- **What**: All 23 raw fits complete (verified per-marker); no correction-step output anywhere. `run_andrillon_report_generation.sh` has an uncommitted edit pointing at `target_confidence`, suggesting intent, but no log/output shows it ran.
- **Why it matters**: Confidence's own neural correlate is unanswered; cross-dimension BH for the other six targets is currently computed with m=6, should be m=7.
- **Scientific proposal**: Run `apply_mcc_postprocessing.py` + `generate_summary_report.py` for confidence, then re-run `omnibus_test.py` unrestricted.
- **Confidence**: CONFIRMED

### [SCI-004] PRDS diagnostic stale relative to the merged 19-marker sleep family — Severity: HIGH
- **Type**: Statistical justification gap
- **Location**: `config_andrillon.yaml` multiple_comparisons comment block (evoked m=4, state m=16, sleep m=5 table)
- **What**: The quoted correlation table describes families that no longer exist as such — `state`'s spectral/connectivity/IT markers were folded into `sleep`, making the active family m=19, for which no correlation table has ever been recomputed anywhere in the repo (confirmed by search).
- **Why it matters**: This table is the entire quantitative justification for BH-over-BY on the family carrying the headline 17/19 result. Direction is inferable (merging in more compositional markers should worsen, not improve, negative dependence) but not verified.
- **Scientific proposal**: Recompute mean r / % negative pairs / % r<−0.3 directly from the 19 markers' already-saved t-maps.
- **Confidence**: CONFIRMED (no recomputation exists); LIKELY (conclusion still holds).

### [SCI-005] Two of onoff's 17 significant markers are the family's most heteroscedastic/kurtotic and remain untransformed — Severity: MEDIUM
- **Location**: `assumption_diagnostics/assumption_summary.csv` (onoff); `response_transform.enabled: false`
- **What**: `psd_relative_gamma` (exkurt 20.5, 100% heteroscedastic) and `slowwaves_Density` (exkurt 7.3, 100% heteroscedastic) are the two most extreme markers in the family, both already flagged by the pipeline's own transform rule, both in the significant list, transform still off.
- **Scientific proposal**: Run the transformed track end-to-end (not just observed t-map correlation) and compare cluster p-values/BH outcome.
- **Confidence**: CONFIRMED diagnostics; LIKELY re: consequence.

### [SCI-006] BH-vs-BY sensitivity analysis never run — Severity: MEDIUM
- **What**: Confirmed absent by search, despite the config's own mandate to report it.
- **Scientific proposal**: Re-run Stage 4 with `fdr_by` against already-computed raw p-values (no re-fit needed) for onoff and valence_sq at minimum — the direct quantitative answer to the user's "too many positives" worry.
- **Confidence**: CONFIRMED

### [SCI-007] Omnibus/assumption-diagnostics scripts overwrite rather than append — Severity: LOW
- **What**: Both write one combined CSV reflecting only the last invocation's scope. Currently non-biting for the omnibus test (2026-07-27 run happened to cover all 6 targets) but biting for assumption diagnostics (`assumption_summary.csv` = onoff only).
- **Scientific proposal**: Always invoke unrestricted when updating the combined file; add a shrink-warning guard.
- **Confidence**: CONFIRMED

### [SCI-008] No persisted `used_config.yaml` makes SCI-001's root cause unverifiable — Severity: MEDIUM
- **What**: Formula gained valence_sq/time_sq terms in a commit dated 2026-07-28, yet target_valence_sq/target_time_sq results date from 2026-07-24 — only possible if disk state differed from git at fit time. No config snapshot exists in any of the 7 target directories.
- **Scientific proposal**: Persist a resolved-config snapshot per model directory on first save.
- **Confidence**: CONFIRMED (observation); LIKELY re: SCI-001 link.

### [SCI-009] Ad-hoc untracked SLURM scripts bypass documented orchestration — Severity: LOW
- **What**: `run_andrillon_marker_array_conf.sh`/`_evoked.sh` (untracked, byte-identical to the committed template) plus custom-named logs (`cbpt_conf_*`, `cbpt_evoked_<predictor>_*`) indicate a manual workaround, likely for the documented SLURM congestion.
- **Scientific proposal**: Delete or formalize (e.g. add a `--family` filter to `submit_parallel_andrillon_markers.sh`) once confidence's MCC step is done.
- **Confidence**: CONFIRMED existence; LIKELY causal link.

### [SCI-010] README §9.2 "broken tests" claim is stale — resolved in code, not docs — Severity: LOW
- **What**: `test_permutation.py`/`test_clustering.py` (importing nonexistent modules) were deleted in commit `4997a7a`; replaced by working tests at repo-root `tests/test_omnibus_test.py` (calibration-under-null, tie-handling, cross-dimension BH monotonicity — using the pipeline's real numbers as fixtures) and `tests/test_residual_shape_diagnostics.py`/`test_response_transforms.py`. `Stats_andrillon/tests/README.md` documents this accurately; the main README does not.
- **Confidence**: CONFIRMED

### [SCI-011] Minor code-quality items — Severity: LOW
- `run_andrillon_pipeline.py` wraps the whole pipeline call in `try/except Exception` (soft violation of the "no try/except" rule; benign, re-raises via `sys.exit(1)`) with a bare `except: pass` on tempfile cleanup. `plot_cbpt_summary_figure.py` hardcodes hex colors instead of `color_palette.yaml`.
- **Confidence**: CONFIRMED

### [Previously known, re-verified, still current] QA exclusion filter declared but not wired — Severity: MEDIUM
- **What**: Re-confirmed by grep: zero references to `qa_summary`/`qa_exclusion`/`exclude_failed_qa` in `andrillon_pipeline.py`. 0/42 subjects excluded by QA regardless of failures.
- **Scientific proposal**: Wire it in, mirroring `Statistics/run_pipeline.py`, alongside the other recommended re-runs.
- **Confidence**: CONFIRMED

---

## 4. Missing Analyses

| # | Missing Analysis | Why It Matters | Priority |
|---|------------------|----------------|----------|
| M1 | Re-run MCC + omnibus for the 6 processed targets against current evoked fits | SCI-001 | High |
| M2 | Run MCC + report + omnibus for `confidence` (all 7 dims) | SCI-003 | High |
| M3 | Recompute PRDS correlation table for the current 19-marker sleep family | SCI-004 | High |
| M4 | Run BH-vs-BY sensitivity for sleep family (onoff, valence_sq) | SCI-006 | High |
| M5 | Fix + re-run `plot_cbpt_summary_figure.py` | SCI-002 | High |
| M6 | Run assumption diagnostics for the other 5 targets + confidence | SCI-005/completeness | Medium |
| M7 | Run transformed-response track for psd_relative_gamma, slowwaves_Density end-to-end | SCI-005 | Medium |
| M8 | Wire QA exclusion filter | Previously known, still open | Medium |
| M9 | Persist config snapshot per results directory | SCI-008 | Medium |
| M10 | Update README §9 tests section; reconcile duplicate array scripts | Hygiene | Low |

---

## 5. Scientific Proposals

**SCI-001**: Treat every evoked-family number as provisional until MCC/omnibus are re-run against the 2026-07-28/29 fits; recompute rather than eyeball against the old BH threshold, since the direction of change was toward more significance in the cells checked.

**SCI-002**: Rebuild `DIMS` programmatically from the actual predictor list and `get_model_folder_name()`; switch the significance criterion to the BH-corrected marker p-value with the existing hierarchical gate.

**SCI-003**: Run the standard post-processing chain for `target_confidence`, then re-run the omnibus test unrestricted so cross-dimension correction spans all 7 dimensions.

**SCI-004**: Recompute the marker-correlation table directly from the t-maps already saved in each `results.pkl` for the actual active sleep family — cheap, no re-fitting needed.

**SCI-005**: Run the transformed track end-to-end for the two flagged markers and compare cluster p-values/BH outcomes, not just observed t-map correlation.

**SCI-006**: Re-run Stage 4 with `fdr_by` against already-computed cluster p-values for onoff and valence_sq — the most direct quantitative answer to "too many positive results."

**M6/M7**: Extend assumption diagnostics to all targets so a flagged marker can be checked across dimensions.

**M9**: Have `save_results()` persist a config snapshot per model directory on first write, to prevent a repeat of the SCI-008 ambiguity.

---

## 6. Prior Report Comparison

This is the first review of the `Stats_andrillon/` pipeline. A prior report (`reports/report_pipeline_review_2026-06-05.md`) exists for a different pipeline (LOSO classification) and contains no applicable findings.

---

## 7. Figures Inventory

| Figure | Path | What It Shows | Scientific Assessment |
|--------|------|---------------|----------------------|
| Paper 2 main figure | `results/figures/cbpt_summary_figure.{png,pdf}` | Topomaps (4 markers × 4 dims) + heatmap (all markers × 6 dims) | **Problematic** — dated 2026-06-09; generating script's folder names don't resolve and significance criterion is uncorrected (SCI-002) |
| Per-marker topomap/cluster/t-dist plots | `results/andrillon_cluster/*__target_*/<marker>/results_*.{png,svg}` | Observed t-map, cluster boundaries, permutation null | Valid for sleep-family; **stale for evoked-family** across all 6 targets (SCI-001) |
| `raw_topographies.pdf` | per marker | Raw topography by predictor split, pre-model | Valid, unaffected by staleness |
| `assumption_summary.png` | onoff only | Skew/kurtosis/outlier bars per marker | Valid for onoff; missing elsewhere (M6) |
| `SUMMARY_TOPOPLOTS_*.pdf` | per target | Post-MCC topoplot summary | Stale for evoked family in all 6 targets; valid for sleep |

---

## 8. Overall Assessment

**What this pipeline found**: A three-tier hierarchy of MW-dimension neural encodability, corroborated at two independent statistical levels for the sleep family: onoff and valence_sq concentrated/localizable (cross-dimension BH≈0.001, peaking in PE_beta); valence/time/time_sq diffuse but real (BH≈0.052); selfother weak/null (p≈0.085). The omnibus test directly answers the user's "is 17/19 a dependence artifact" worry: built from the same permutations with no independence assumption, it reaches the same conclusion (observed count 17 vs null mean 1.90, p=0.0002).

**Confidence in findings**: High for the sleep family of the six processed dimensions. Low, currently, for the evoked family of those same dimensions (SCI-001) and not yet assessable for confidence (SCI-003).

**Biggest threats to validity, ranked**: (1) unverified evoked-family conclusions with at least one qualitative shift (SCI-001); (2) non-functional/methodologically-wrong main figure (SCI-002); (3) stale PRDS justification + missing BY sensitivity check on the very family carrying the headline result (SCI-004/006); (4) confidence's hierarchy placement and the correctly-sized 7-dimension correction not yet computable (SCI-003).

**Recommended next steps**:
1. Re-run MCC + omnibus for the six processed targets to pick up the evoked re-fit; specifically check whether valence_sq/P3a survives correction.
2. Complete the confidence target (MCC, report, fresh 7-dimension omnibus).
3. Recompute the PRDS diagnostic for the current sleep family and run the BH-vs-BY sensitivity check — the quantitative answer to the question that motivated this review.
4. Fix and regenerate the Paper 2 main figure with correct paths and BH-corrected significance.
5. Then extend assumption diagnostics to all seven targets, wire the QA filter, and persist config snapshots going forward.
