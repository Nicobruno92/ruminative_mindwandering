# EEG Mind-Wandering Analysis — Claude Instructions



Virtual Environment Use. You never have to create a new one, there is an specific environment designed for each kind of task. You have to activate the appropriate one before running.

basic stuff => base
crating plots specific tasks => plots
anything related to machine learning => ML
anything related to natural languange processing => nlp
for anything related to eeg analysis => eeg
for junifer specific tasks => junifer

## Paper Structure

This project is split into two papers:

### Paper 1 — Patients & Behavior
Focuses on patient group differences and subjective/objective reporting of mind-wandering. More behavioral, less EEG-heavy.

### Paper 2 — EEG
Focuses on neural correlates of mind-wandering independent of patient status.

**Central question**: Does EEG encoding of MW depend on which dimension you measure, and is that encoding strong enough for individual-level prediction?

**Paper structure (high level):**

**Section 1 — Behavior (brief)**
- LMM: probe dimensions (onoff, valence, selfother, time, confidence) predict performance markers (errors, RTCV, omissions)
- Establishes construct validity of the probe dimensions
- Results: `results/Behavior/objective_markers/lmm_probe_dimensions/`

**Section 2 — CBPT: dimension-specific neural signatures**
- Andrillon pipeline (LMM + cluster-permutation): which EEG markers show group-level correlates for each dimension?
- **Linear specification only** (as of 2026-08-13). A quadratic variant adding
  orthogonalised `valence_sq`/`time_sq` was previously the reported one; those
  terms have been **removed from every analysis in the project** — see
  "Quadratic Terms: Removed" below. Five targets: onoff, valence, selfother,
  time, confidence.
- **All Section 2 numbers below predate that removal and are being recomputed.**
  They came from the quadratic specification, in which the quadratic covariates
  measurably distorted the linear `valence`/`time` estimates (sign flips at
  several electrodes). Do not quote any tier, p-value, or marker from this
  section until the linear-only re-run lands.
- **Multiple comparisons are read at two levels** (`Stats_andrillon/`, see the
  `andrillon-mcc-fdr-per-family` memory for the full method):
  1. *Marker-wise* — one max-statistic p per marker, Benjamini-Hochberg within each
     marker family (evoked m=4, sleep m=19). Answers "which specific marker?".
  2. *Omnibus* (`Stats_andrillon/omnibus_test.py`) — a family-level permutation test.
     Answers "does this dimension leave ANY neural trace?", which the marker-wise
     step cannot: signal spread over several moderate markers fails every
     marker-wise test yet is collectively far from chance. Two statistics: a *count*
     (markers below α vs the permutation null) and a *min-p* (best marker, FWER-valid
     under arbitrary dependence — the assumption-free counterpart to BH). Both are
     then BH-corrected across the **five** dimensions within each family (was six
     when the quadratic targets existed).
- **SUPERSEDED — the three-tier hierarchy below came from the quadratic
  specification and is retained only so nobody re-derives it from stale output.**
  Two of its tiers rested on terms that no longer exist, and `valence`/`time`
  were estimated with the quadratic covariates in the model. Recompute all of it
  from the linear-only run before citing anything:
  - ~~**Concentrated AND localizable** — `onoff` and `valence_sq`, both peaking in
    PE-beta.~~ `onoff`/PE-beta is the one part independently verified to survive
    (robust to dropping the quadratics *and* to a leverage drop-test);
    `valence_sq` is gone.
  - ~~**Distributed but real** — `time`, `valence` (linear), `time_sq` at BH≈0.052.~~
  - ~~**Weak/null** — `selfother` (omnibus p≈0.085).~~
- The old "48-electrode valence×confidence" and "slow waves increase with valence"
  claims do **not** survive the corrected analysis — do not reuse them.
- Main figure: topomaps (representative markers × dimensions) + heatmap (marker
  families × dimensions). Report both marker-wise and omnibus levels; flag the
  distributed tier as suggestive, not confirmatory.
- Results: `results/andrillon_cluster/` (per-dir `multiple_comparisons_summary.csv`,
  `mcc_family_composition.csv`; family-level `omnibus_test.csv` at the root).

**Section 3 — Classification: individual-level decodability**
- Within-subject (WS) and leave-one-subject-out (LOSO) RF classifiers, median split
  per dimension, on a **175-feature** space (the 23 Andrillon CBPT markers per ROI).
- **All numbers below are the 2026-07-30 re-run.** Anything quoting the old
  AUCs (onoff 0.703, valence 0.658, LOSO 0.628, "16/29") predates the
  feature-space cache fix and must not be reused — see
  `memory/classification-canonical-feature-space-177.md`. Never quote the
  convenience `*_summary*.csv` files; recompute with
  `scripts/recompute_headline_numbers.py`.
- **Within subject**, every dimension is decodable (all p_FDR < .05):
  onoff 0.721 (15/29 subjects sig.) > valence 0.647 (5/23) > confidence 0.640
  (7/23) > selfother 0.587 (5/31) > time 0.554 (3/27).
- **LOSO**: only **onoff 0.627** and **confidence 0.620** survive FDR. Valence
  0.536, time 0.538, selfother 0.517 do not.
- The headline is a **dissociation**: valence is second-best within subject yet
  at chance across subjects (gap 0.111), while confidence loses almost nothing
  (gap 0.020). Affective valence during MW has a largely idiosyncratic neural
  signature; metacognitive confidence has a shared one.
- ~48% of subjects are not individually decodable even for onoff — result, not failure.
- Main figure: `scripts/make_fig_section3_decoding.py` →
  `results/figures/section3_decoding/` (per-dimension WS/LOSO dot plot, per-subject
  points, permutation band; fill encodes significance).
- Results: `mw_classification_pipeline/results/MW_Classification/`

**Feature consistency (WS vs LOSO)** — `scripts/feature_consistency_analysis.py`,
figures via `scripts/make_fig_feature_consistency.py`, outputs in
`results/feature_consistency/`:
- Per subject, mean(|SHAP|) from their own model vs from the model trained
  without them, over the same trials.
- **Group level**: Spearman ρ = 0.41 (onoff), 0.35 (valence), 0.31 (confidence),
  0.22 (selfother), 0.003 (time, n.s.) over 175 features. Collapsing collinear
  ROI columns to the 23 markers raises this to **0.58 (onoff) / 0.56 (valence) /
  0.58 (confidence)**, with selfother 0.31 (p = .07) and time ≈ 0 both
  non-significant. (The valence² 0.39 / time² 0.39 entries were dropped with the
  quadratic contrasts — see "Quadratic Terms: Removed".) Part of the feature-level disagreement is
  arbitrary choice among collinear ROI columns, not disagreement about markers.
- **Marker aggregation must be `mean` (per ROI), never `sum`** — set in
  `scripts/config_feature_consistency.yaml`. The 4 evoked markers own 1 column
  each and the 19 sleep markers own 9, so summing hands sleep markers 9× the
  mass before any data is seen: it buries P3b (which rises to 4th of 23 under
  `mean`) and inflates every marker-level ρ, because the ROI-count pattern is
  identical in both pipelines and correlates with itself. The superseded
  sum-based figures reported ρ = 0.43–0.87; do not reuse those.
- **Individual level**: mean per-subject ρ ≤ 0.08; only 6/29 (onoff), 4/23
  (valence), 2/23 (confidence) subjects individually significant. Split-half
  noise ceiling ≥ 0.97, so this is not seed/CV noise (it does *not* bound
  single-subject sampling noise — state that limitation when citing it).
- The old "WS and LOSO pick opposite features, r ≈ −1" claim is a
  **selection-on-extremes artifact** of correlating over a top-10 ∪ top-10
  subset. Do not reuse it, and never annotate a correlation computed on a
  top-N subsample.

**Narrative bridge between sections 2 and 3:**
> CBPT establishes which dimensions leave a detectable group-level neural trace. Classification asks whether that trace is strong enough to predict MW state in a specific person.

**Caveat on that bridge**: do not assert that the two orderings validate each
other. The CBPT side is being recomputed under the linear-only specification
(see Section 2), so there is currently no CBPT tier ranking to compare against.
The previous claim — that CBPT's localizable tier was `onoff` + `valence_sq`
while LOSO picks `onoff` + `confidence` — is void on the `valence_sq` half.
Re-check both orderings against the new numbers before writing this bridge.

---

## SART Task Design

**Paradigm**: Sustained Attention Response Task (SART) — go/no-go.
**Software**: MATLAB + Psychophysics Toolbox 2017b.
**Setup**: BenQ XL2430t 24" 1080p screen, ~70 cm from participant's eyes, dark grey background.

### Stimuli & Timing

| Element | Duration |
|---------|----------|
| Digit display (1–9, white) | 800–1100 ms (variable) |
| Fixation cross (ISI) | 900–1200 ms (variable) |

- **Go**: press right-index mouse button for any non-target digit.
- **No-go** (inhibit): withhold response when the target digit appears.
- **Target mapping**: digit **3** → Sart1 & Sart2 ; digit **5** → Sart3 & Sart4.

### Block & Trial Structure

| Per SART | Value |
|----------|-------|
| Total trials | 450 |
| Blocks | 15 (pseudo-random mix of 20-, 30-, 40-trial blocks) |
| Probes per SART | 15 (one at block end) |
| Total probes across 4 SARTs | 60 |

**Target frequency per block** (pseudo-randomised):
- 20-trial block → 1–2 targets
- 30-trial block → 2–3 targets
- 40-trial block → 3–4 targets

**Target constraints**:
- Minimum spacing: 5 trials between consecutive targets.
- No target in the **last 8 trials** before a probe (avoids re-engaging attention immediately pre-probe).

### Thought Probes — MDES

Multi-Dimensional Experience Sampling (MDES); continuous horizontal slider (0–100) for each question.

| Question | Poles | Dimension label |
|----------|-------|-----------------|
| Q1 | Completely on-task ↔ Not at all on-task | `onoff` |
| Q2a (order randomised) | Centered on self ↔ Not centered on self | `selfother` |
| Q2b | Negative ↔ Positive | `valence` |
| Q2c | Past-oriented ↔ Future-oriented | `time` |
| Q3 | Confidence in self-assessment (low ↔ high) | `confidence` |

- Q1 presented on screen 1; Q2 (3 sub-questions, random order) on screen 2; Q3 on screen 3.
- Scale convention: **0 = off-task / negative / past / self-focused / low confidence**, **100 = on-task / positive / future / other-focused / high confidence**.
- These raw scores map directly onto the event-label scale in the event string (`onoff99`, `valence47`, etc.).

---

## Critical Rules (Always Apply)

1. **NEVER hardcode paths** → use `utils/bids_compliance.py`
2. **NEVER hardcode parameters** → load from `config.yaml`
3. **Subject IDs**: zero-padded strings `"02"` to `"43"` (no `"01"`)
4. **Tasks**: `["Sart1", "Sart2", "Sart3", "Sart4"]` (case-sensitive)
5. **Pre-probe analysis**: filter distance `-5` to `-1`
6. **No `try/except` blocks** in scientific scripts — let errors surface; fix root causes
7. **No magic numbers** — all constants go to `config.yaml` or top of script
8. **Explicit random seeds** — always use `random_state` from config, never implicit defaults

## Essential Imports

```python
import yaml
from utils.bids_compliance import read_epochs, save_evokeds, load_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

config = yaml.safe_load(open('config.yaml'))
```

## Data Locations

```
_RAW_DATA/{sub}/                          → Raw BrainVision
BIDS/raw/{sub}/eeg/                       → BIDS raw
BIDS/derivatives/{sub}/eeg/               → Preprocessed (*_desc-autoPreproc_epo.fif)
BIDS/features/{sub}/eeg/
  ├── junifer/                            → Spectral/connectivity (*_markers.pkl)
  └── mne_evokeds/                        → ERPs (*_desc-probe-{N}_{label}_ave.fif)
results/{pipeline}/                       → Outputs (gitignored)
```

## Project Structure

```
<repo>/
├── BIDS/                          # BIDS compliant data (raw & derivatives)
├── Preprocessing_pipeline_new/    # EEG Preprocessing modules
├── ERPs_new/                      # ERP analysis modules
├── Statistics/                    # Statistical analysis (LMMs, etc.)
├── junifer_markers/               # Feature extraction pipeline (spectral/conn)
├── mw_classification_pipeline/    # Machine Learning and classification
├── Behavior/                      # Behavioral analysis and dashboards
├── utils/                         # Shared project utilities
├── results/                       # Generated outputs (gitignored)
└── tests/                         # Automated tests
```

## Config Locations

```
Preprocessing_pipeline_new/config.yaml    → ICA, artifacts
ERPs_new/config.yaml                      → ROIs, baseline
Statistics/config.yaml                    → LMM, thresholds
junifer_markers/*/config.yaml             → Markers, bands
```

## Standard Workflow

```python
# Load & filter
epochs = read_epochs("04", "Sart1", config['project']['derivatives_root'])
epochs = filter_epochs_by_distance_to_probe(epochs, distance=5)  # -5 to -1

# Classify on/off-task
classified = classify_onoff_epochs(epochs, split='median')
evoked_on  = classified['high'].average()
evoked_off = classified['low'].average()

# Save
save_evokeds([evoked_on, evoked_off], "04", "Sart1",
             config['project']['features_root'],
             desc="probe-015", labels=['onTask', 'offTask'])
```

## Event Structure

Format: `go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`

- **Dimensions**: 0–100 scale (0=off-task, 100=on-task)
- **Distance**: negative=before probe, positive=after
- **Use**: `-5` to `-1` for mental state prediction

## Run Commands

```bash
# Single subject
python Preprocessing_pipeline_new/preprocessing_pipeline.py \
  --config config.yaml --subject 04 --task Sart1

python ERPs_new/make_probe_evokeds.py \
  --config config.yaml --subject 04 --task Sart1

# Batch (SLURM)
sbatch Preprocessing_pipeline_new/run_preprocessing_slurm.sh
sbatch ERPs_new/run_complete_erp_pipeline.sh

# Junifer (sequential)
cd junifer_markers/1.markers_h5_creation && sbatch slurm_array_junifer.sh
cd ../2.h5_to_pkl && sbatch batch_convert_h5_to_pkl_parallel.sh
cd ../3.aggregate_probes && sbatch run_aggregate_slurm.sh
```

## SLURM Array Index

`array_idx = subject_idx * 4 + task_idx`

Example: Sub-04 (idx=2), Sart2 (idx=1) → `2*4+1 = 9`

## Key Functions

```python
# I/O
read_epochs(subject, task, derivatives_root)
save_evokeds(evokeds, subject, task, features_root, desc, labels)
load_evokeds(subject, task, features_root, desc, labels)

# Analysis
filter_epochs_by_distance_to_probe(epochs, distance)  # keeps -N to -1
classify_onoff_epochs(epochs, split='median')          # {'high': on, 'low': off}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check zero-padding (`"04"`), case (`"Sart1"`), BIDS functions |
| No epochs match | Verify distance filter, check `*_desc-autoPreproc_epo.fif` exists |
| Memory error | Add `gc.collect()`, use `max_files_per_batch` parameter |
| Missing config param | Check correct config file, use `config['project']['key']` path |

---

## Scientific Guardrails

- **Context-First**: Before generating code, read `README.md` to understand the project structure. Do not guess file paths.
- **Epistemic Humility**: Flag high-risk statistical decisions (thresholding, p-values) and ask for scientific justification. Do not default to `p < 0.05` without context.
- **Hallucination Check**: Only import libraries listed in `pyproject.toml` or `environment.yml`. If a new library is needed, propose adding it to the environment file first.
- **BIDS Compliance**: Never write to `data/raw`. Validate entity ordering in filenames. When creating `.tsv`/`.csv` output, suggest a corresponding `.json` sidecar.
- **Scientific Self-Evaluation**: Before declaring a task finished, review generated plots/stats for physiological plausibility (ERP latencies, frequency band power distributions).
- **Determinism**: Scientific pipelines must be fully deterministic. No conditional fallbacks. Optional steps controlled by config flags, not exception catching.
- **Garden of Forking Paths**: If exploring multiple parameters or thresholds, document all of them and report result sensitivity. Flag exploratory findings as *exploratory*, not *confirmatory*.
- **Anti-HARKing**: Do not rewrite the goal of a script to match what the data showed. Always distinguish planned analysis from post-hoc discoveries.

---

## Color Palette

Single source of truth: `color_palette.yaml` (repo root). **Never hardcode hex values in plots** — load this file and key into it, so every figure stays consistent. Load it path-relative to the script (`Path(__file__).resolve().parents[N] / "color_palette.yaml"`), not via cwd.

**Base palette (Observable10):**

| Name | Hex | | Name | Hex |
|------|-----|--|------|-----|
| Blue | `#4269D0` | | Pink | `#FF8AB7` |
| Orange | `#EFB118` | | Purple | `#A463F2` |
| Red | `#FF725C` | | Light Blue | `#97BBF5` |
| Cyan | `#6CC5B0` | | Brown | `#9C6B4E` |
| Green | `#3CA951` | | Gray | `#9498A0` |

**Per-dimension assignment (stable across all figures):**

| Dimension | Color | Hex | | Dimension | Color | Hex |
|-----------|-------|-----|--|-----------|-------|-----|
| `onoff` | red | `#FF725C` | | `confidence` | orange | `#EFB118` |
| `valence` | blue | `#4269D0` | | `pc1` | cyan | `#6CC5B0` |
| `selfother` | green | `#3CA951` | | `pc2` | pink | `#FF8AB7` |
| `time` | purple | `#A463F2` | | `pc3` | light blue | `#97BBF5` |

Neutral: gray `#9498A0` = permutation/chance baseline and non-dimension covariates (e.g. `time_on_task`); accent (default single-series) = blue `#4269D0`.

**Significance encoding (project-wide convention):** color encodes the *dimension*, never significance. Significance is encoded by **fill**: significant (after correction) = **filled/solid** marker; non-significant = **hollow/empty** (white face, colored edge) or **dashed**. Keep this consistent across forest plots, topomaps, heatmaps, and any new figure — so a reader can read dimension from hue and significance from fill independently.

Currently wired into the Paper 2 plots: `mw_classification_pipeline/` and `results/Behavior/objective_markers/lmm_probe_dimensions/` (generators in `Behavior/Objective_Markers/lmm_probe_dimensions.py`). Apply the same palette to any new plot.

---

## Dimension Order (Figures)

Canonical left-to-right / top-to-bottom order for any multi-dimension figure —
panel order, heatmap/table columns, forest-plot rows, legend order:

```
onoff → valence → selfother → time → confidence
```

That is the complete order — the quadratic terms that used to be interleaved
after their linear parents were removed from the project (see "Quadratic Terms:
Removed"), so no interleaving rule is needed any more.

**Why this order and not another**: it matches `color_palette.yaml`'s
`dimensions:` key order, which was already the majority convention across the
codebase before this was written down. It is **not** derived from the SART
probe's on-screen question order (Q1 onoff, Q2a selfother, Q2b valence, Q2c
time, Q3 confidence — see "SART Task Design" above puts selfother before
valence) and **not** from decodability or effect-size ranking (Section 3's
AUC ranking is onoff > valence > confidence > selfother > time; CBPT's tiers
in Section 2 are a different ordering again). Keeping the panel layout itself
independent of any one result keeps the figure from priming the reader toward
a conclusion before they read the fill/significance encoding — rankings are
reported as findings inside prose and tables, not baked into panel position.

Wired into (as of 2026-08-12):
- `color_palette.yaml` — `dimensions:` key order
- `Stats_andrillon/plot_paper_figures.py` — `HEATMAP_DIMENSION_ORDER`
- `Behavior/Probe_analysis/probe_dimension_cloud_plot.py` — `DIMENSIONS`, `CORR_VARS`
- `mw_classification_pipeline/scripts/generate_combined_classification_figure.py`
  — `DIMENSIONS`, `GROUP_ROW_DIMENSIONS`
- `mw_classification_pipeline/scripts/config_feature_consistency.yaml` —
  `contrasts[]`

**Former exception, now retired**: `Stats_andrillon/plot_paper_figures.py`'s
`SECONDARY_DIMENSIONS` used to be ordered by CBPT tier (concentrated →
distributed → null) on the grounds that the ordering *was* the finding. It was
reverted to canonical order on 2026-08-13 because those tiers came from the
quadratic specification, which has been retired — baking a superseded ranking
into panel position primes the reader toward a conclusion the current data has
not re-established. Do not restore a tier ordering until the linear-only re-run
is in and the tiers have actually been recomputed. If any figure is ever
narrative-ordered again, document the deviation next to its order list rather
than letting it drift silently.

---

## Dimension Labels & Pole Wording (Figures)

Two different pieces of text appear per dimension across Paper 2 figures, and
each has exactly one canonical wording — don't introduce a synonym for either,
even under space pressure (shorten by line-wrapping or omitting a word, not by
substituting a different word).

**1. Display label** (panel title, legend entry, table row, heatmap column
header before any width-driven wrapping):

| Dimension | Canonical label | | Dimension | Canonical label |
|-----------|------------------|--|-----------|------------------|
| `onoff` | `On/Off-Task` | | `time` | `Time` |
| `valence` | `Valence` | | `confidence` | `Confidence` |
| `selfother` | `Self/Other` | | | |

Line-wrapping to fit a narrow column is fine (`"On/Off-\nTask"`,
`"Self/\nOther"`) as long as the unwrapped text is still the canonical label —
`Self vs. other` / `Self / other` / `Self/other` were three such variants
that existed in the repo and have been consolidated to `Self/Other`;
`On/off task` / `On/Off Task` / `On/off-task` were consolidated to
`On/Off-Task` (sentence-case prose mentioning the dimension inline, e.g. "the
on/off-task effect", is not a label and is exempt).

**2. Pole / extreme wording** (what the low and high end of the 0–100 scale
are called — axis endpoint labels, "higher when X" effect-direction
annotations, correlation-plot axis names):

| Dimension | Low pole (0) | High pole (100) |
|-----------|--------------|------------------|
| `onoff` | `off-task` | `on-task` |
| `valence` | `negative` | `positive` |
| `selfother` | `self-focused` | `other-focused` |
| `time` | `past` | `future` |
| `confidence` | `low` | `high` |

This is the short form (`SHORT_POLES` in `Stats_andrillon/plot_paper_figures.py`
— the first place it was defined). A full-sentence variant exists for CBPT
effect-direction annotations only (`POLE_LABELS` in the same file, e.g.
`"higher when OFF-task\n(mind-wandering)"` / `"higher when ON-task"`) — use it
only where a bare pole word would be ambiguous about what "higher" means;
everywhere else (raw-data axis labels, correlation matrices) use the short
form above. `self` / `other` is *not* valid shorthand for `self-focused` /
`other-focused` — it was found as an inconsistent shortening in
`probe_dimension_cloud_plot.py` and corrected (2026-08-12).

`confidence`'s row was corrected a second time (2026-08-13): the 2026-08-12
pass had flagged `low` / `high` as an inconsistent shortening too and
"corrected" it to `unconfident` / `confident` — backwards. The SART Task
Design section above defines Q3's own poles literally as "(low ↔ high)", and
`unconfident` carries connotations (anxiety, insecurity) the slider never
measured — it only asked how confident the self-assessment was. `low` / `high`
reads correctly wherever it's used because the column header or panel title
right next to it always already says "Confidence".

Wired into (as of 2026-08-13):
- `Stats_andrillon/plot_paper_figures.py` — `POLE_LABELS`, `SHORT_POLES`,
  `HEATMAP_COLUMN_LABELS`, `SECONDARY_COLUMN_LABELS`
- `Behavior/Probe_analysis/probe_dimension_cloud_plot.py` — `DIMENSIONS`
  (`label`, `pole_low`, `pole_high`), `CORR_VARS`
- `Behavior/Objective_Markers/lmm_probe_dimensions.py` — `PREDICTOR_LABELS`
- `mw_classification_pipeline/scripts/generate_combined_classification_figure.py`
  — `DIMENSIONS[].label`
- `mw_classification_pipeline/scripts/config_feature_consistency.yaml` —
  `contrasts[].label`

---

## Quadratic Terms: Removed (2026-08-13)

`valence_sq` / `time_sq` — the orthogonalised `(x-50)²/50` curvature terms — were
removed from **every** analysis: CBPT (`Stats_andrillon`, `Statistics`,
`Statistics_connectivity`), Behaviour (`lmm_probe_dimensions.py`), and the
classification contrasts. Do not reintroduce them without changing how the term
is constructed; the problem is the construction, not a bug.

**Why.** The probe sliders are bounded and skewed, so the orthogonalised residual
has skew 2.4–2.7 and its top 5% of observations carry ~60% of its variance (vs
13% for a linear predictor like `onoff`). Every effect built on it was carried by
that tail:

| test | full sample | dropping top-5% leverage |
|---|---|---|
| CBPT `time_sq`, evoked/P1 at Oz | t = 2.76 | **t = 1.15** |
| Behaviour `valence_sq` → omission_rate | z = 2.52, p_FDR .047 | **z = −1.10** (sign flip) |
| Behaviour `valence_sq` → total_errors | z = 2.25, p_FDR .048 | **z = −0.73** (sign flip) |
| control: CBPT `onoff`, PE-beta at FC5 | t = −5.24 | t = −5.13 (robust) |

Model-free binning showed the supposed U was a single tail bin (n = 28 for time)
with the adjacent bin going the other way. Extremes are unbalanced by
construction — valence has 19 probes ≤10 vs 615 ≥90, so the "extreme negative"
arm rested on 19 probes.

**Two traps to avoid if this ever comes up again:**
- *Orthogonalisation is not the culprit and re-fitting it does not help.* The
  highest-order term is **invariant** under that reparametrisation — verified: t
  is identical raw vs orthogonalised vs re-orthogonalised on the analysis sample.
  What the quadratic term did change was the **linear** term, whose t swung from
  −0.43 to +3.05 on the same data depending on parametrisation. That is why
  `valence` and `time` also had to be re-estimated, not just the `_sq` targets.
- *`min_predictor_variability_sq: 15` was justified by a false premise* ("quadratic
  predictors live on a 0-50 scale, use half the linear threshold"). The
  orthogonalised residual spans ≈ −10..+79, not 0-50. Since variability is measured
  as within-subject range, that criterion selected subjects **for having extreme
  probes** (r = 0.53 with max|time−50|), i.e. it enriched for the very leverage
  driving the effect.

**Classification is the exception, and was not itself invalid**: the median split
only uses rank order, so the long tail is just "high". Those contrasts were
disabled for consistency, not because they were wrong. If ever revived, note the
"extreme" class is confounded with the raw dimension (corr +0.27 with valence,
+0.15 with time), so it partly re-learns the linear dimension.

---

## Quadratic-Dimension Display Labels *(historical)*

The `valence_sq` / `time_sq` terms were removed from every analysis on
2026-08-13 (see "Quadratic Terms: Removed"), so these labels are no longer
wired into anything. Recorded only so that old figures and CSVs remain
readable:

| Column key | Display label |
|------------|---------------|
| `valence_sq` | `Neutral/Emotional` |
| `time_sq` | `Present/NotPresent` |

Note `Present/NotPresent` was in any case a poor label: the `time_sq`
regressor's minimum sat at time ≈ 63, not 50 ("present"), and it was
asymmetric — +71 at the past extreme vs +17.9 at the future extreme, so a
positive coefficient weighted *past* about 4:1 rather than "not present"
symmetrically. If a curvature construct is ever reintroduced, name it from its
actual fitted shape, not from the intended one.

---

## Marker Naming (Figures)

Canonical stem per marker family — matches the family name shown in
`DISPLAY_GROUPS`/`HEATMAP_GROUP_LABELS` — with a full-word and a compact
rendering:

| Family | Stem | Full-word example | Compact example |
|--------|------|--------------------|------------------|
| Evoked (ERP) | *(none — component name is already unambiguous)* | `P1`, `N1`, `P3a`, `P3b` | same |
| Spectral (relative power) | `PSD` | `PSD alpha` | `PSD α` |
| Complexity / information | `PE` *(Kolmogorov has no stem — the one non-band marker in its family, already unambiguous)* | `PE alpha`, `Kolmogorov` | `PE α`, `KoC` |
| Connectivity (wSMI) | `wSMI` | `wSMI alpha` | `wSMI α` |
| Slow waves | `SW` | `SW density` | `SW density` |

**Exactly two verbosity tiers exist, not more or a mix**: full spelled-out
band names, for the wide topomap/heatmap panels in
`Stats_andrillon/plot_paper_figures.py`'s `MARKER_DISPLAY_NAMES`; and a
Greek-symbol compact tier for
`mw_classification_pipeline/scripts/make_fig_ws_loso_sign_forest.py`'s
`MARKER_LABELS`, which lists dozens of marker×ROI combinations as forest-plot
y-tick labels and needs every character. Both tiers use the *same* stem word
per family — only the band spelling (`alpha` vs `α`) and stem punctuation
(`PSD` vs no equivalent shorthand needed) change between them. Never
introduce a third stem word for a family that already has one — `rel.` was
briefly a synonym for `PSD` in the spectral family (in the compact tier only)
and has been corrected to `PSD` (2026-08-12) so a reader moving between the
CBPT figure and the forest plot sees the same word for the same marker.

**Do not fix**: `Stats_andrillon/plot_cbpt_summary_figure.py` still carries
the old `rel.` wording, but it is an orphaned precursor to
`plot_paper_figures.py` — nothing imports or invokes it (verified by
repo-wide reference search, 2026-08-12). Delete it rather than patch it if it
resurfaces in a future figure pass.

Wired into (as of 2026-08-12):
- `Stats_andrillon/plot_paper_figures.py` — `MARKER_DISPLAY_NAMES`
- `mw_classification_pipeline/scripts/make_fig_ws_loso_sign_forest.py` —
  `MARKER_LABELS`

---

## Figure Assembly (Paper 2)

**Default: Plotly via the `scientific-plots` skill's `sciplot` helper**, not
raw Plotly calls or matplotlib. Invoke the `scientific-plots` skill before
building or restyling any figure — it defines the actual API (`sp.save`,
`sp.make_template`, `sp.mm2px`/`sp.pt2px`, `sp.grid`, `sp.panel_labels`,
`sp.TYPE_PT`); don't re-derive it here. Repo-specific conventions on top of
that skill:
- Figure size is set in **millimetres** (`WIDTH_MM` / `HEIGHT_MM` constants
  at the top of the script, e.g. `Behavior/Probe_analysis/probe_dimension_cloud_plot.py`),
  passed to `sp.save(fig, out_dir, name, width_mm=..., height_mm=...)`, which
  writes both `.svg` and `.png`.
- Colors always resolve through `color_palette.yaml` (see "Color Palette"
  above) via `sp.make_template(pal)`, never a hardcoded hex.
- Multi-panel figures use letter labels (`sp.panel_labels`) rather than
  relying on subplot titles alone to identify panels in prose.
- Output goes to `results/{pipeline}/...` or `results/figures/{section}/...`
  per the "Paper Structure" section above — never inside a source directory.

**Exception — matplotlib, not sciplot**: `Stats_andrillon/plot_paper_figures.py`
(CBPT topomaps/heatmap) uses matplotlib directly, sized in **inches**
(`figsize=`) and saved through `Statistics/plot_results.save_figure_multiformat`
(also PNG+SVG, `dpi=300`), because MNE's topomap plotting (`mne.viz`) only
draws onto matplotlib axes — there is no Plotly equivalent. Keep this the one
documented exception rather than porting it to sciplot or adding a second
undocumented one elsewhere.

**When a figure mixes both** (e.g. a combined panel that embeds a matplotlib
topomap next to Plotly panels), render each half through its native library
and compose the raster/vector outputs at the results-folder level — do not
attempt to fake MNE topomaps in Plotly or reimplement `sciplot`'s layout
logic in matplotlib.

---

## Code Style

- **Docstrings**: Numpy/Scipy style. Every function, class, and module must be documented.
- **Sectioning**: Use `# ===` separators to group Imports / Configuration / Helper Functions / Main.
- **Naming**: Use domain-specific names (`high_onoff_probes`, `marker_intensity`). Avoid `df1`, `data_list`.
- **Type hints**: Required on all new/modified function signatures.
- **Transparency**: If participants/trials are excluded, provide results both with and without exclusions. Document every cleaning step.

---

## Modularity

- **YAML-only config**: Never hardcode parameters in Python or Bash. No double-configurations.
- **Pure Functions**: Prefer functions with no side effects; keep I/O separate from computation.
- **Data-Code Separation**: Raw data is immutable. Path resolution via `utils/bids_compliance.py`.
- **Structure for new features**: Implementation + unit tests in `tests/` + docstrings.
- **Results isolation**: All outputs (plots, CSVs, models) go to `results/{pipeline_name}/`. Never write results into source code directories.

---

## Git Practices

- **Atomic Scientism**: Do not mix scientific changes with cosmetic changes in the same commit.
- **The "Why" Mandate**: Commit messages for parameter/logic changes must explain the scientific rationale.
- **No `git checkout`** to solve development issues or revert changes — use normal edits and commits.
- **Large files**: Never commit files > 10MB. Use DataLad or `.gitignore`.
- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `exp:` (for experiments).

---

## Automation Checklist

Before declaring any pipeline stage complete:
- [ ] Fast local test script exists and passes
- [ ] Config loaded with `yaml.safe_load()`
- [ ] Subject is string: `"04"` not `4`
- [ ] Task case-correct: `"Sart1"` not `"sart1"`
- [ ] BIDS functions used (no hardcoded paths)
- [ ] Pre-probe filter applied for mental state analysis
- [ ] Results folder contains `used_config.yaml`
- [ ] Performance metrics reviewed and documented
- [ ] SLURM scripts match Python entrypoint
- [ ] No binary/large files in `git status`
- [ ] All stochastic operations have explicit `random_state`
- [ ] No `try/except` blocks present
- [ ] Type hints on all new/modified functions
