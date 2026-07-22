# GLMM for Objective Behavioural Markers — Design Spec

**Date:** 2026-07-22
**Target:** `Behavior/Objective_Markers/lmm_probe_dimensions.py` (Paper 2, Section 1 — Behaviour)
**Status:** Approved design, pending implementation plan

---

## 1. Problem

The LMM pipeline that models objective behavioural markers (`omission_rate`,
`commission_rate`, `total_errors`, `rtcv`) as a function of probe dimensions
fits a **Gaussian** mixed model via `statsmodels.mixedlm` and **never validates
a single distributional assumption**. `fit_lmm()` returns the tidy coefficient
table directly; no residuals are extracted, and there is no QQ plot, normality
test, homoscedasticity test, or influence diagnostic anywhere in the module or
in `Behavior/Objective_Markers/`.

This matters because the response variables violate Gaussian assumptions
severely, not marginally. Measured on the `full_segment` dataset (n = 2460
probes, 42 subjects):

| Marker | Range | Skew | % zeros |
|---|---|---|---|
| `omission_rate` | [0, 1] | 6.08 | 65.2 % |
| `commission_rate` | [0, 1] | 1.59 | 66.1 % |
| `rtcv` | (0.071, 2.715] | 6.65 | 0 % |

The binomial denominators are the crux:

| Dataset | `n_go` (omission denom.) | `n_nogo` (commission denom.) |
|---|---|---|
| `full_segment` | 14–37 (median 27) | **1–4 (median 3)** |
| `n10` | 9–10 | **0 or 1** |

`commission_rate` in `full_segment` is a proportion over **1 to 4 trials**,
taking values such as 0, ⅓, ½, 1. In `n10` it is **binary**, and 847 of 2460
probes have `n_nogo = 0`, making the rate undefined. A Gaussian LMM on a binary
outcome is indefensible, and it can predict impossible values outside [0, 1].

### Why response transformation alone is insufficient

Applying logit or arcsine to a proportion with denominator 2 requires an
arbitrary continuity constant, and on a binary variable it is meaningless.
Transformation is therefore appropriate **only for `rtcv`** (continuous,
strictly positive, skew 6.65). The rate markers require a model that treats
the underlying counts as counts.

---

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| GLMM engine | R `lme4::glmer` 2.0.1 via `Rscript` subprocess | Only correctly-specified option available; see §3 |
| Primary model | GLMM | The Gaussian and transformed fits become pre-registered sensitivity analyses |
| `total_errors` | Redefined as `(n_omissions + n_commissions) / n_trials_window` | The current sum-of-two-rates has mismatched denominators (~27 vs ~3) and is a proportion of nothing |
| Moderation analysis | Also GLMM | Mixing estimands within one paper section is not defensible |

### Consequences that change published output

Two consequences are deliberate and must be reflected in the manuscript:

1. **Coefficients change units.** β moves to **log-odds** (binomial) and **log**
   (Gamma). Every forest plot axis, scatter-grid annotation, and the Section 1
   text must be updated accordingly.
2. **`total_errors` changes definition.** The new joint proportion weights
   errors by natural trial frequency (~27 go vs ~3 no-go trials), so it is
   dominated by omissions, whereas the old sum-of-rates weighted omission and
   commission equally. These answer different questions. The old definition is
   retained in the Gaussian sensitivity track for continuity.

---

## 3. Engine selection: why `Rscript` subprocess, not `rpy2`

The environment constrains this decision. Findings from investigation:

- No `pymer4`, `rpy2`, `bambi`, or `pymc` installed in any conda env.
- `conda.anaconda.org` is **blocked by the institutional proxy**
  (`proxy-icm:3128`) — connection fails with any CA bundle. PyPI and CRAN are
  reachable.
- `pip install rpy2` **fails to build** against the conda R
  (`_get_r_cmd_config` → `IndexError`).
- Installing `lme4` from CRAN into the conda R **fails to compile**: the conda
  R `Makeconf` lacks `SHLIB_LIBADD`, breaking `nloptr`'s configure step.
  `R CMD config` does not expose arbitrary Makeconf variables, so patching
  `Makeconf` does not fix it.
- **`lme4` 2.0.1 is already installed** in the Lmod modules `R/4.6.0` and
  `R/4.4.3`. No compilation required.

Calling R through a subprocess therefore avoids all linking fragility, works
with the module system, and leaves the R model code as a versioned,
reviewable artefact in the repository. Verified working: binomial `glmer` and
`Gamma(link="log")` `glmer` both fit successfully on the real data via
`Rscript` under `R/4.6.0`.

`lmerTest` is **not** required: it provides Satterthwaite degrees of freedom
for `lmer`, whereas `glmer` reports Wald z-tests natively, and the Gaussian
sensitivity track stays on statsmodels.

---

## 4. Model specification

### Primary GLMM

| Marker | Response | Family | Link |
|---|---|---|---|
| `omission_rate` | `cbind(n_omissions, n_go - n_omissions)` | binomial | logit |
| `commission_rate` | `cbind(n_commissions, n_nogo - n_commissions)` | binomial | logit |
| `total_errors` | `cbind(n_om + n_com, n_trials_window - n_om - n_com)` | binomial | logit |
| `rtcv` | `rtcv` (strictly positive) | Gamma | log |

Data validity was verified across both datasets: no numerator exceeds its
denominator, and `n_go + n_nogo == n_trials_window` in every row. In `n10`, the
847 probes with `n_nogo = 0` drop out of the commission model naturally, which
is a further argument for the binomial formulation over an ad-hoc rate.

Fixed effects, random effects, and predictor preprocessing are **unchanged**
from the current model:

```
~ onoff + valence + valence_sq + selfother + time + time_sq
  + confidence + time_on_task + (1|subject)
```

Predictors remain z-scored (`STANDARDIZE_PREDICTORS = True`), and the quadratic
terms remain globally orthogonalized against their linear counterparts. Random
slopes are explicitly out of scope; the random-intercept structure is preserved
so the GLMM is directly comparable to the existing model.

### Overdispersion — rule fixed a priori

Overdispersion is expected with aggregated binomial counts. To avoid a
data-dependent specification choice (a garden-of-forking-paths violation of the
project guardrails), the rule is fixed in advance:

- The Pearson χ²/df dispersion statistic is **always computed and reported**,
  for every model in every track.
- The observation-level random effect (OLRE) variant is **always fitted** as a
  pre-registered sensitivity analysis for the three **binomial** markers.
- Both are always reported regardless of the dispersion value. No model is
  selected on the basis of its dispersion statistic.

OLRE applies to the binomial markers only; it is not fitted for `rtcv`, where
the Gamma family carries its own dispersion parameter. It is additionally not
identifiable when the denominator is 1 (`n10` `commission_rate`), so it is
skipped there and the skip is recorded explicitly in the output rather than
left as a silently missing row.

### Sensitivity track: transformed response

- `rtcv` → `log(rtcv)`
- Rate markers → empirical logit with Haldane correction:
  `log((y + 0.5) / (n - y + 0.5))`, which is the standard treatment for small
  denominators.

Fitted with the existing `statsmodels.mixedlm` machinery.

### Sensitivity track: Gaussian

The current model, unchanged in family and estimator.

### Which `total_errors` definition each track uses

To keep `model_comparison.csv` a like-for-like comparison of *specification*
rather than a confound of specification and marker definition, **all three
tracks model the same responses**, using the new joint-proportion
`total_errors`.

The original definition is fitted **once**, in the Gaussian track only, and
written as a separate marker named `total_errors_legacy`. Its sole purpose is
continuity with already-published output; it is excluded from
`model_comparison.csv` because it is not comparable to the other rows.

---

## 5. Architecture

Three new modules plus one config file. **No refactoring of existing code**
beyond the integration points listed in §6.

### `Behavior/Objective_Markers/glmm_fit.R`

Reads a model-data CSV and a JSON spec (formula, family, response
construction, optimizer), fits `glmer`, and writes two CSVs: the coefficient
table and a diagnostics row. Pure computation and I/O at the boundaries; no
plotting.

### `Behavior/Objective_Markers/glmm_backend.py`

Python wrapper around the R subprocess. **The central design constraint: it
returns exactly the same tidy schema as `fit_lmm`**, column for column:

```
predictor, estimate, std_error, t_value, p_value,
conf_lower, conf_upper, p_fdr, significant_fdr
```

so every existing plotting function (`plot_combined_forest`,
`plot_scatter_grid`, `plot_combined_panel`, `_draw_forest_panel`) works
unchanged.

`t_value` holds the Wald **z** statistic that `glmer` reports. The name is kept
because the plotting code reads `row["t_value"]` directly; renaming it would
require touching every plotting function for no scientific gain. A `z_value`
column is added as an explicit duplicate so that the CSVs on disk are not
misleading about what the statistic actually is, and the axis/annotation labels
are updated to say *z* where the GLMM track is plotted.

Three further columns are appended for provenance: `converged`, `dispersion`,
and `n_obs`.

Public surface:

```python
fit_glmm(data, marker, spec, predictors) -> pd.DataFrame
fit_moderation_glmm(data, marker, moderator, spec) -> dict
```

`fit_moderation_glmm` mirrors the return contract of `fit_moderation_lmm`
(keys: `marker`, `moderator`, `interaction_term`, `estimate`, `std_error`,
`t_value`, `p_value`, `n_obs`, `n_subjects`, `converged`) so
`run_moderation_analysis` can dispatch between backends without changing its
downstream FDR and plotting logic.

### `Behavior/Objective_Markers/diagnostics.py`

Residual diagnostics for both tracks — the gap that motivated this work:

- Gaussian / transformed: QQ plot, residuals vs fitted, skew and kurtosis of
  residuals, Breusch-Pagan test.
- GLMM: Pearson and deviance residuals, dispersion statistic, and **binned
  residual plots** (Gelman & Hill) for the binomial models. `DHARMa` is not
  installed and cannot be installed (conda channel blocked, and it is not
  present in the R modules), so simulation-based residuals are out of scope;
  binned residuals are the standard alternative for binomial GLMs.

Note on testing residual normality: with n = 2460, Shapiro-Wilk rejects on
trivial deviations. Diagnostics are therefore reported as **effect-size-like
summaries plus plots**, not as pass/fail hypothesis tests.

### `Behavior/Objective_Markers/glmm_config.yaml`

Per the project's YAML-only rule, nothing is hardcoded: marker→family mapping,
response-column construction, transformation definitions, R module name and
version, optimizer, iteration limits, dispersion settings, and output subdirectory
names all live here.

---

## 6. Integration points

The count columns (`n_omissions`, `n_go`, `n_commissions`, `n_nogo`,
`n_trials_window`) already survive the merge in `run_pipeline_for_dataset` and
pass unchanged through `standardize_predictors`, which only touches predictor
columns. `df_fit` inside `_fit_predictor_set` therefore already carries
everything `glmer` needs — no data-plumbing changes are required.

| Location | Change |
|---|---|
| `run_pipeline_for_dataset` (~L1836) | Compute `total_errors` as the new joint proportion (used by all three tracks) and `total_errors_legacy` as the old sum-of-rates (Gaussian track only) |
| `_fit_predictor_set` (~L1749) | Loop over the three tracks instead of one; dispatch to `fit_glmm` or `fit_lmm` per track |
| `run_moderation_analysis` (~L1598) | Accept a backend parameter; dispatch to `fit_moderation_glmm` or `fit_moderation_lmm` |
| Plotting functions | **Unchanged** — guaranteed by the tidy-schema contract |

FDR correction logic is unchanged and applies within each track independently:
across predictors within a model for the additive analysis, and across all
marker × moderator tests for the moderation analysis.

---

## 7. Outputs

```
results/Behavior/objective_markers/lmm_probe_dimensions/<dataset>/
  glmm/                     ← PRIMARY
  sensitivity_gaussian/     ← existing output, unchanged
  sensitivity_transformed/
  sensitivity_olre/
  diagnostics/
  model_comparison.csv
  used_config.yaml
```

Each track directory keeps the current file layout (`<marker>_lmm_results.csv`,
`lmm_summary_all_markers.csv`, `moderation_summary.csv`, and the plot suite).

`model_comparison.csv` is the central transparency artefact: one row per
marker × predictor, with the estimate, p-value, and FDR significance in each of
the three tracks, so the manuscript can state explicitly whether conclusions are
robust to specification — or report exactly where they are not.

Per the project checklist, each dataset directory receives a `used_config.yaml`
snapshot.

---

## 8. Testing

Unit tests in `tests/test_glmm_objective_markers.py`:

1. **Count consistency** — numerators never exceed denominators; `n_go + n_nogo
   == n_trials_window`; both datasets.
2. **Empirical logit** — Haldane-corrected transform matches hand-computed
   values, including at y = 0 and y = n.
3. **Tidy-schema conformance** — `fit_glmm` output has exactly the columns the
   plotting functions read, with matching dtypes.
4. **R round-trip** — synthetic data with a known injected log-odds effect;
   assert `glmer` recovers it within tolerance. This validates the whole
   subprocess path (serialisation, formula construction, parsing) end to end.
5. **`n10` commission edge case** — probes with `n_nogo = 0` are dropped and the
   reported `n_obs` matches.

Test 4 is the critical one: it is the only check that the Python↔R boundary
does not silently corrupt the model.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `glmer` convergence failures on sparse binomial interactions (commissions over 1–4 trials) | `bobyqa` optimizer with raised iteration limit; **convergence status recorded as a column in every results CSV**. Non-converged models are reported, never silently dropped |
| R module unavailable on a compute node | Module name/version in config; the backend asserts `lme4` availability and fails loudly at startup rather than mid-run |
| Subprocess masking R errors | Return code checked explicitly; R `stderr` propagated into the Python exception message. Consistent with the project's no-`try/except` rule — errors surface rather than being swallowed |
| Runtime growth (3 tracks × 4 markers × 2 datasets, plus 7 moderators → ~200 fits) | Models are small; measured `glmer` fits complete in seconds. If needed, the R script accepts several models per invocation to amortise startup |

---

## 10. Explicitly out of scope

- Random slopes (`(1 + onoff|subject)`) — the random-intercept structure is
  preserved for comparability.
- Beta-binomial models — would require `glmmTMB`, which is not installed and
  cannot be installed given the blocked conda channel.
- Simulation-based residuals (`DHARMa`) — same constraint; binned residuals
  serve instead.
- Refactoring the module-level constants of `lmm_probe_dimensions.py` into YAML.
  The existing hardcoded constants violate the project's YAML-only rule, but
  fixing that is unrelated to this work; only the new functionality is
  config-driven.
- Re-running or altering any EEG/CBPT or classification pipeline.
