# LOSO: Median Position/Variability vs Decodability — Design Spec

**Date:** 2026-06-10
**Paper:** Paper 2 — EEG (Section 3, exploratory follow-up)
**Status:** Exploratory (not part of confirmatory analysis plan)

---

## Context

Section 3 reports a per-dimension LOSO decodability hierarchy (mean AUC across
subjects): onoff > confidence ≈ valence > time ≈ selfother. This is an
exploratory follow-up asking *why* that hierarchy might look the way it does,
at two levels:

1. **Dimension level** — is a dimension's group-level LOSO decodability
   related to how *consistent* subjects' within-subject medians are for that
   dimension? (Low across-subject variability in medians → the
   `within_subject_median` binarization is more "comparable" across subjects
   → a model trained on N-1 subjects' class definitions may generalize better
   to the held-out subject.)

2. **Subject level (within each dimension)** — is a *subject's* LOSO AUC
   related to where their own median sits on the 0-100 scale? Hypothesis:
   subjects whose median is close to 50 (the scale midpoint) produce a more
   balanced, behaviorally cleaner low/high contrast and may be more
   decodable; subjects with a median near an extreme push more values into
   one class, potentially making the contrast noisier.

These two questions are related to but distinct from the existing
`plot_auc_vs_onoff_dispersion` (per-subject SD of ratings vs AUC, already
computed for all 5 dimensions). That plot asks "how widely does this subject
use the scale?"; this analysis asks "where is this subject's median?" and
"how similar are subjects' medians to each other, per dimension?".

Findings here are descriptive/exploratory and will be reported as such — no
claim of confirmatory significance, especially for Fig A (n=5 dimensions).

---

## Data Sources

| File | Used for |
|------|----------|
| `results/Behavior/probe_data/probe_level_aggregated_data.csv` | Per-probe values of onoff/valence/selfother/time/confidence, per subject |
| `mw_classification_pipeline/results/MW_Classification/LOSO/{contrast}/all/rf/used_config.yaml` | `_data_provenance.subjects_final` (subject set actually used in that dimension's LOSO) and `label_contrasts.{contrast}.label_source` (column name) |
| `mw_classification_pipeline/results/MW_Classification/LOSO/{contrast}/all/rf/rf_loso_100runs_loso_subject_metrics.csv` | Per-subject LOSO `auc` (= "decodability of that participant") |

The 5 contrasts: `ON_vs_OFF_within_median` (onoff), `valence_within_median`,
`selfother_within_median`, `time_within_median`, `confidence_within_median`.

For each dimension, restrict to that dimension's `subjects_final` (different
N per dimension: 29/27/32/28/24) — same set used to compute the AUCs, so the
two quantities being compared come from the same sample.

**Per-subject median**: `df.groupby("subject")[label_source].median()` on
`probe_level_aggregated_data.csv`, restricted to `subjects_final`.

---

## Fig A — Dimension-level (`dimension_auc_vs_median_variability`)

- 5 points, one per dimension.
- x = SD across subjects of the per-subject median (computed above), in
  rating-scale units (0-100).
- y = mean LOSO AUC across `subjects_final` for that dimension (from
  `rf_loso_100runs_loso_subject_metrics.csv`).
- Each point labeled with the dimension name.
- No regression line / r,p (n=5 too small to be meaningful) — purely
  descriptive scatter, axis ranges include AUC=0.5 chance line for reference.

## Fig B — Subject-level, faceted (`auc_vs_median_distance_from_50_faceted`)

- 5 panels, one per dimension (onoff, valence, selfother, time, confidence).
- x = `abs(subject_median - 50)` — distance of the subject's median from the
  scale midpoint.
- y = subject's LOSO `auc` for that dimension.
- Per panel: scatter with subject ID labels, OLS regression line, Pearson
  r/p, chance line at AUC=0.5 — same visual style as
  `plot_auc_vs_onoff_dispersion` (plotly, `COLORS[0]`, `plotly_white`).
- Faceting via plotly subplots in a 2x3 grid (5 panels + 1 empty cell),
  shared y-axis range across panels for comparability.

---

## Output

New folder:
```
mw_classification_pipeline/results/MW_Classification/LOSO/median_position_analysis/
├── dimension_auc_vs_median_variability.png/.pdf/.html
├── dimension_auc_vs_median_variability_data.csv      # dimension, median_sd, mean_auc, n_subjects
├── auc_vs_median_distance_from_50_faceted.png/.pdf/.html
├── auc_vs_median_distance_from_50_data.csv           # dimension, subject, subject_median, dist_from_50, auc
└── used_config.yaml                                   # data sources, methodology note (exploratory), per-dimension subjects_final/N
```

## Script

New standalone script: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- No CLI args needed (paths are fixed relative to repo structure); reuses
  `COLORS` and plotly styling conventions from `utils/plotting_utils.py`.
- Pure functions for: loading per-dimension subject medians + AUCs, building
  Fig A data, building Fig B data, plotting each figure.
- Type hints, numpy-style docstrings, no try/except (let missing files error
  loudly — all 5 dimensions' LOSO results already exist on disk).
