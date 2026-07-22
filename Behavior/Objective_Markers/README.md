### Objective behavioural markers (Behavior/Objective_Markers)

Models how the MDES probe dimensions (`onoff`, `valence`, `selfother`, `time`,
`confidence`) predict objective behavioural markers from the SART — omission
rate, commission rate, total errors and RT variability. This is **Paper 2,
Section 1 (Behaviour)**: it establishes construct validity of the probe
dimensions before the EEG sections.

---

## Which script does what

| File | Role |
|---|---|
| `aggregate_objective_markers_by_probe.py` | Upstream. Turns trial-level SART data into one row per probe: `n_go`, `n_nogo`, `n_omissions`, `n_commissions`, `rt_mean`, `rt_sd`, `rtcv`. Produces the two input CSVs. |
| `objective_markers_analysis.py` | Descriptive analysis of the markers (distributions, group comparisons). |
| `vtc_analysis.py`, `vtc_zone_analysis.py` | Variance Time Course: "in the zone" / "out of the zone" classification from RT variability. Independent of the LMM/GLMM track. |
| **`lmm_probe_dimensions.py`** | **Main entry point.** Fits every model track, runs the moderation analysis, and generates all figures and tables. |
| `glmm_config.yaml` | All GLMM parameters: marker→family map, response columns, transforms, Rscript path, optimizer. Nothing is hardcoded in Python. |
| `response_transforms.py` | Pure functions — binomial response construction, Haldane-corrected empirical logit, log transform. No I/O, no R. |
| `glmm_fit.R` | Fits one `lme4::glmer`, writes coefficients + diagnostics + residuals as CSV. |
| `glmm_backend.py` | Python↔R boundary. Returns results in the **same tidy schema as `fit_lmm`**, so all plotting code works unchanged. |
| `diagnostics.py` | Residual diagnostics: QQ, residuals vs fitted, binned residuals, Breusch-Pagan. |

---

## How to run

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/plots/bin/python \
  Behavior/Objective_Markers/lmm_probe_dimensions.py
```

Use the **`plots`** env, not `eeg`: `eeg` has an incompatible plotly/kaleido
pair (5.18 + 1.0) that breaks static image export. Tests run under `eeg`
(`plots` has no pytest):

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Regenerate figures without refitting: `--plots-only`.

---

## Why GLMM and not a plain LMM

The pipeline originally fitted a **Gaussian** LMM and validated no
distributional assumption. The responses violate those assumptions severely:

| Marker | Range | Skew | % zeros |
|---|---|---|---|
| `omission_rate` | [0, 1] | 6.08 | 65 % |
| `commission_rate` | [0, 1] | 1.59 | 66 % |
| `rtcv` | (0.07, 2.72] | 6.65 | 0 % |

The denominators are the crux. `commission_rate` is a proportion over **1–4
no-go trials** in `full_segment`, and is **binary** in `n10` (847 of 2460
probes have no no-go trial at all). A Gaussian model on a binary outcome is
indefensible and can predict values outside [0, 1]. Transformation cannot fix
this — a logit over a denominator of 2 needs an arbitrary continuity constant.
The counts must be modelled as counts.

Full rationale: `docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md`.

## Model specification

| Marker | Response | Family | Link |
|---|---|---|---|
| `omission_rate` | `cbind(n_omissions, n_go - n_omissions)` | binomial | logit |
| `commission_rate` | `cbind(n_commissions, n_nogo - n_commissions)` | binomial | logit |
| `total_errors` | `cbind(n_om + n_com, n_trials - n_om - n_com)` | binomial | logit |
| `rtcv` | `rtcv` (> 0) | Gamma | log |

Fixed effects and random effects are unchanged from the original model:

```
~ onoff + valence + valence_sq + selfother + time + time_sq
  + confidence + time_on_task + (1|subject)
```

Predictors are z-scored; quadratic terms are globally orthogonalized against
their linear counterparts (the `poly(x, 2)` parametrization).

**Coefficients are on the link scale**: log-odds for the binomial markers,
log for `rtcv`. They are *not* in rate units.

**`total_errors` was redefined** as `(n_omissions + n_commissions) /
n_trials_window`. The previous definition summed two rates with different
denominators (~27 go vs ~3 no-go trials), which is not a proportion of
anything. The old definition survives as `total_errors_legacy` in the Gaussian
track only, for continuity with published output.

## Why four tracks

Every track is fitted and reported **regardless of outcome**. Choosing a
specification after seeing the results would be a garden-of-forking-paths
violation, so the rule is fixed in advance:

- `glmm` — **primary**. Binomial / Gamma.
- `sensitivity_olre` — same binomial model plus an observation-level random
  effect, `(1|subject) + (1|obs_id)`. Absorbs overdispersion. **Not a different
  family** — the plain binomial forces variance to be exactly `n·p·(1−p)`, and
  when the data vary more than that the standard errors come out too small.
- `sensitivity_gaussian` — the original misspecified model, kept as a baseline.
- `sensitivity_transformed` — Gaussian LMM on `log(rtcv)` / empirical logit.

## R dependency

`glmer` runs as an `Rscript` **subprocess**, not through `rpy2`. On this
network `conda.anaconda.org` is blocked by the institutional proxy, `rpy2`
fails to build against the conda R, and `lme4` cannot compile there — but
`lme4` 2.0.1 is already installed in the `R/4.6.0` module. The absolute binary
path in `glmm_config.yaml` works without `module load`.
