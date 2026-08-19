# Statistical Analysis — Behaviour (Paper 2, Section 1)

Draft Methods subsection. Numbers verified against
`results/Behavior/objective_markers/lmm_probe_dimensions/` (run of 2026-07-23).

---

## Statistical analysis

All analyses were conducted at the probe level, with probes nested within
participants (42 participants, 2,460 probes; 15 probes per SART block × 4
blocks). Generalized linear mixed models (GLMMs) were fitted in R 4.6.0 using
`lme4` 2.0.1, called from Python 3.12.9; Gaussian sensitivity models were fitted
with `statsmodels` 0.14.6.

### Choice of error distribution

The objective behavioural markers are not continuous unbounded variables, and
were modelled accordingly rather than with a Gaussian linear mixed model. Error
markers are counts of events out of a known number of trials: omissions out of
go trials (median 27, range 14–37) and commissions out of no-go trials (median
3, range 1–4). Because the no-go denominator is small, the commission
*proportion* takes only a handful of discrete values per probe, and in the
10-trial pre-probe window it is binary. Treating such a variable as Gaussian
assumes a variance structure the data cannot have and permits predictions
outside [0, 1].

We therefore modelled the three error markers as **binomial** with a logit
link, using the trial counts directly as `cbind(events, trials − events)`:

- **Omissions**: `n_omissions` out of `n_go`
- **Commissions**: `n_commissions` out of `n_nogo`
- **Total errors**: `n_omissions + n_commissions` out of all trials in the
  pre-probe window

Total errors were defined as a single proportion of all trials rather than as
the sum of the two error rates, because those rates have different denominators
(≈27 go vs ≈3 no-go trials) and their sum is therefore not a proportion of any
well-defined quantity. Probes contributing no trials to a given denominator
carry no information about the corresponding proportion and drop out of that
model; this removed 847 probes from the commission model in the 10-trial window
(n = 1,613 of 2,460), and none in the full-block window.

Response-time variability (RTCV, the within-window coefficient of variation of
correct-trial RTs) is strictly positive and right-skewed, with a variance that
grows with the mean. It was modelled as **Gamma** with a log link. Probes with
no valid RTs were excluded (n = 2,445 of 2,460 in the full-block window).

### Model structure

Each marker was modelled as a function of the five MDES probe dimensions plus
time on task, with a by-participant random intercept:

```
marker ~ onoff + valence + valence² + selfother + time + time²
         + confidence + time_on_task + (1 | subject)
```

Quadratic terms for valence and time were included to allow curvature beyond a
linear effect, and were centred at the scale midpoint and orthogonalized
against their own linear term by pooled ordinary least squares — the `poly(x,
2)` parametrization. This leaves the quadratic estimate, standard error and
p-value unchanged while removing its collinearity with the linear term. All
predictors were z-scored before fitting, so coefficients are expressed per
standard deviation of the predictor. **Coefficients are on the link scale**:
log-odds for the binomial models and log for the Gamma model.

Models were fitted by maximum likelihood with the BOBYQA optimizer. Fixed
effects were tested with Wald z-tests. Convergence status, singularity
(`isSingular`), random-effect standard deviations and the maximum absolute
gradient were recorded for every model and are reported alongside the
estimates.

### Multiple comparisons

For the additive models, Benjamini-Hochberg FDR correction was applied across
the eight predictors within each model. For the moderation analysis, correction
was applied across all marker × moderator tests jointly. The threshold was
α = 0.05 throughout.

### Moderation

To test whether the effect of on-task focus on behaviour depends on the other
probe dimensions, we fitted `marker ~ onoff × moderator + (1 | subject)` for
each marker and each of the remaining dimensions plus time on task, using the
same error distribution as the corresponding additive model. The
`onoff × moderator` interaction was the test of interest.

### Sensitivity analyses

The choice of model family was fixed before inspecting the results, and three
sensitivity analyses were specified in advance and are reported regardless of
their outcome:

1. **Gaussian linear mixed models** on the raw markers, the specification these
   data would conventionally receive.
2. **Transformed responses** fitted with Gaussian linear mixed models:
   log-transformed RTCV, and Haldane-corrected empirical logit
   (`log[(y + 0.5)/(n − y + 0.5)]`) for the error proportions, which remains
   finite when no events are observed.
3. **Observation-level random effects (OLRE)** added to the binomial models,
   `(1 | subject) + (1 | observation)`, to accommodate overdispersion.

Overdispersion was quantified as the Pearson χ²/df ratio. Because
overdispersion is expected in aggregated binomial counts, the OLRE variants
were fitted for all binomial markers irrespective of the observed dispersion,
so that no specification was selected after seeing the data.

### Model diagnostics

Residual behaviour was assessed for every model in every specification.
Because the sample size (n ≈ 2,460) makes formal normality tests reject on
negligible deviations, residual distributions are summarised descriptively
(skewness, kurtosis) and displayed as quantile-quantile and residual-versus-
fitted plots, with heteroscedasticity assessed by the Breusch-Pagan test. For
the binomial models, where raw residuals are uninformative against fitted
values, we additionally computed binned residual plots (Gelman & Hill) with
±2 SE bands.

---

## Notes for the Results section (not paper text)

Points that follow from the analysis and should be stated explicitly:

**Dispersion and what survives it.** Omissions (χ²/df = 1.95) and total errors
(1.73) were overdispersed; commissions were not (0.92). Under OLRE the
coefficients are essentially unchanged but standard errors roughly double, and
five effects in the full-block window lose significance: `time`, `time²` and
`time_on_task` on omissions, and `time²` and `time_on_task` on total errors.
These should not be claimed. The `onoff` effect survives in every marker,
window and specification.

**A singular OLRE fit.** The OLRE was singular for commissions in both windows
(observation-level variance = 0), consistent with their dispersion below 1;
coefficients were identical to the plain binomial to four decimal places. This
is worth one sentence: it indicates there is no overdispersion to absorb, not a
model failure.

**Specification matters.** Agreement between the Gaussian and GLMM
significance calls was 75% (full-block) and 84% (10-trial). Notably,
`onoff → RTCV` was **not** significant under the Gaussian model but is under
the Gamma GLMM (β = −0.041, p_FDR < .001) — the misspecified model missed a
real effect. Conversely, `time²` and `valence²` on commissions were significant
only under the Gaussian model.

**Residual improvement, honestly stated.** GLMM residual skewness is markedly
lower than Gaussian (RTCV 2.13 vs 5.15; omissions 2.64 vs 4.37; total errors
2.30 vs 4.04), but Breusch-Pagan remains significant for most GLMM models. The
respecification improves fit substantially without producing textbook
residuals, and should be described that way.

**Replication across windows.** `onoff` replicates for all four markers.
`time_on_task → RTCV` replicates. The quadratic terms do not replicate across
windows and should be treated as exploratory. `selfother` was not significant
for any marker in any window or specification.
