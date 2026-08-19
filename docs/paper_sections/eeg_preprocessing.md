# EEG Preprocessing — Methods

Draft Methods subsection. Every parameter below was read from
`Preprocessing_pipeline_new/config.yaml` and the pipeline source; every number
was recomputed from
`/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives/qa_summary.csv`
(state of the derivatives on 2026-07-27). Section "Discrepancies to resolve" is
*not* paper text — it lists things that must be fixed or decided before these
methods are submitted.

---

## EEG acquisition and data organisation

EEG was recorded with a 64-channel EasyCap montage (CACS-64, extended 10/20
placement) plus two bipolar electro-oculogram channels (VEOG, HEOG), digitised
at 500 Hz with an online hardware band-pass of 0.016–250 Hz and a 50 Hz line
frequency. Each participant completed four SART runs (`Sart1`–`Sart4`) of
approximately 20 min (median recording duration ≈ 1,198 s). Forty-two
participants (IDs 02–43) contributed 168 runs.

Raw BrainVision recordings were converted to BIDS in a separate harmonisation
step (`data_harmonization/`), which (i) applied the CACS-64 montage, (ii)
recoded the experimental triggers into a single self-describing annotation per
trial carrying the trial type, accuracy, the five MDES probe scores, the
distance in trials to the upcoming probe and the probe index (e.g.
`go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`),
(iii) inserted a `THOUGHT_PROBE` annotation at each probe onset and a `BAD_rest`
annotation spanning each probe-and-rest period (from probe onset to the start of
the next block), and (iv) resampled the data to 250 Hz. All subsequent
preprocessing operated on these BIDS-formatted recordings at 250 Hz; no further
resampling was applied.

## Preprocessing

Preprocessing was fully automated and deterministic (fixed random seed 42 for
every stochastic step), implemented in MNE-Python and run independently for each
subject × run.

**Removal of non-task periods.** Segments annotated `BAD_rest` — the interval
between the onset of a thought probe and the beginning of the next block, during
which participants answered the MDES questions — were excised from the
continuous recording before any other operation, by cropping the good intervals
and concatenating them. This prevents the probe-answering period, which contains
mouse movement and unconstrained gaze, from contaminating artifact-subspace
calibration, bad-channel detection and the ICA solution. Because these segments
begin at probe onset, the pre-probe windows used for the state analyses are not
affected.

**Line noise.** A zero-phase FIR notch filter was applied at 50 and 100 Hz.

**Artifact Subspace Reconstruction.** High-amplitude transients were attenuated
with ASR (`meegkit`, Euclidean method, cutoff 3 SD, 0.5 s sliding windows with
0.66 overlap, maximum channel-dropout fraction 0.1). The covariance model was
calibrated on the first min(120 s, 30 %) of each recording.

**Bad-channel detection.** Noisy, flat and outlier channels were identified with
PyPREP's `NoisyChannels.find_all_bads` (RANSAC and channel-wise criteria,
detrending enabled). Detection was run on a working copy that was band-pass
filtered 1–30 Hz and pre-cleaned with a fast, discarded FastICA decomposition
(95 % of variance, 200 iterations, 1–40 Hz), from which ocular components
(correlation with VEOG/HEOG) and muscle components (`find_bads_muscle`,
threshold 0.7) were removed. This step exists solely to prevent frontal
electrodes from being flagged as broken because of slow drifts and blinks; the
detected channel labels were then transferred to the unfiltered recording.

**Interpolation, montage and reference.** Channels flagged as bad were
interpolated with spherical splines (`mode="accurate"`) and the bad list was
reset. The CACS-64 montage was applied and the data were re-referenced to the
average of the 64 scalp electrodes (applied directly, not stored as a
projector); EOG channels were typed as `eog` and therefore excluded from the
reference.

**Filtering.** Two streams were derived from the referenced recording with
zero-phase FIR filters: a decomposition stream band-passed 1–30 Hz, used only to
fit ICA, and an analysis stream band-passed 0.1–35 Hz, which carries all data
used downstream.

**Independent component analysis.** ICA was fitted on the decomposition stream
with the Picard algorithm (all components up to the data rank, 1,000 maximum
iterations, `random_state=42`). Components were labelled with ICLabel
(`mne-icalabel`), and each component received an artifact score equal to its
maximum ICLabel probability across the four artifact classes retained (*muscle
artifact*, *eye blink*, *heart beat*, *channel noise*). Converging evidence from
template matching added a bonus of 0.30 to that score when the component
correlated with the EOG channels and 0.30 when it was flagged by MNE's muscle
detector (threshold 0.85); components labelled *channel noise* received 0.60.
Scores were capped at 1. A component was excluded when its score reached a
variance-adjusted threshold, defined as 0.60 plus 0.5 times the fraction of
total variance that the component explained (capped at 0.95), so that a
component carrying a large share of the signal had to be identified as an
artifact with correspondingly higher confidence. The unmixing solution was then
applied to the analysis stream. Runs in which the excluded components jointly
accounted for more than 90 % of the total variance were flagged for quality
control (see below); no component was retained on the basis of that budget.

**Epoching.** Three epoch sets were extracted from the cleaned continuous data:

| Set | Anchor | Window | Baseline | Purpose |
|-----|--------|--------|----------|---------|
| `evoked` | go and no-go stimuli | −0.2 to 1.0 s | −0.2 to 0 s | ERP analyses |
| `state` | thought probe onset | 10 s pre-probe, split into 3 s mini-epochs with 2 s overlap (1 s step, 8 per probe) | none | Spectral / connectivity markers of mental state |
| `sleep` | thought probe onset | 20 s pre-probe, one window per probe | none | Sleep-like markers (slow waves, complexity) |

ERP epochs were restricted to go/no-go events, rejected if they overlapped a
`BAD_*` annotation, and screened with a deliberately permissive peak-to-peak
threshold of 500 µV intended to catch only gross electrode failures. State
epochs inherited the full probe description (all five MDES scores and the probe
index) in their event label, so that each mini-epoch remains traceable to the
probe it precedes and to its temporal distance from it. Repeated events were
dropped.

**Epoch-level artifact rejection.** ERP and state epochs were passed through
AutoReject (cross-validated over 10 folds, Bayesian-optimisation threshold
search, candidate interpolation counts {1, 2, 4, 8, 16}, consensus grid
{0.5 … 1.0}, `random_state=42`), which interpolates locally noisy channels
within an epoch and rejects epochs that cannot be repaired. Sleep epochs were
not submitted to AutoReject, since a 20 s window would be rejected for a
transient that occupies a negligible fraction of it; they are cleaned by ICA
only.

**Outputs.** For each subject × run the pipeline wrote the ICA-cleaned
continuous recording (`desc-icaClean_eeg.fif`) and the three epoch sets
(`desc-evoked_epo.fif`, `desc-state_epo.fif`, `desc-sleep_epo.fif`), restricted
to the 64 EEG channels, together with an MNE quality-control report per run.
Current source density transformation is implemented but was disabled
(`use_csd: false`).

## Quality control and exclusions

Each run was scored on a fixed set of criteria: proportion of interpolated
channels (threshold 0.20), proportion of epochs rejected by AutoReject
(threshold 0.35), proportion of epochs dropped because of annotations
(threshold 0.50), the ICA variance budget (0.90), and the requirement that at
least one component be removed when EOG channels are present. A run failed
quality control if it exceeded the bad-channel or epoch-rejection thresholds,
exceeded the variance budget, or removed no component despite available EOG.

Of the 168 possible runs, 166 completed with full outputs; two runs (sub-12
`Sart4`, sub-38 `Sart4`) produced ERP epochs but no pre-probe epochs and no
quality-control record, and are therefore unavailable for the state and sleep
analyses. Across the 166 recorded runs:

| Metric | Median | Mean | Range |
|--------|--------|------|-------|
| Interpolated channels (of 64) | 4.5 | 5.3 (8.3 %) | 0–27 |
| ICA components removed | 15 | 14.6 (25.6 % of components) | 3–28 |
| Variance removed by ICA | 62.8 % | 59.8 % | — |
| ERP epochs retained | 421 | 410.4 | 80–433 |
| State epochs retained (max 120) | 105.5 | 99.3 | 21–120 |
| Sleep epochs retained (max 15) | 15 | 14.8 | 3–15 |

AutoReject removed 2.5 % of constructed ERP epochs and 5.2 % of state epochs.
Sixteen runs (9.6 %) failed quality control: ten for exceeding the bad-channel
threshold and six for exceeding the ICA variance budget.

## Software

Python 3.10, MNE-Python 1.9.0, `mne-icalabel` 0.7.0, `python-picard` 0.8,
`pyprep` 0.4.3, `autoreject` 0.4.3, `meegkit` 0.1.9, NumPy 2.2.6, SciPy 1.15.2.
All parameters are stored in `Preprocessing_pipeline_new/config.yaml`; the
pipeline is run per subject × run and was executed as a 168-element SLURM array
(`run_preprocessing_slurm.sh`).

## Parameter reference

| Stage | Parameter | Value |
|-------|-----------|-------|
| Sampling | Acquisition / analysis rate | 500 Hz → 250 Hz (harmonisation) |
| Notch | Frequencies | 50, 100 Hz (FIR) |
| ASR | cutoff / win_len / overlap / dropout / method | 3.0 SD / 0.5 s / 0.66 / 0.1 / euclid |
| ASR calibration | window | min(120 s, 30 % of run), ≥ 10 s |
| PyPREP | RANSAC / channel-wise / detrend | on / on / on |
| PyPREP working copy | band-pass; throwaway ICA | 1–30 Hz; FastICA 95 % var, 1–40 Hz, muscle thr 0.7 |
| Reference | Type | Average of 64 scalp electrodes |
| Filters | ICA stream / analysis stream | 1–30 Hz / 0.1–35 Hz, zero-phase FIR |
| ICA | method / n_components / max_iter / seed | Picard / data rank / 1,000 / 42 |
| IC selection | base ICLabel threshold | 0.60 |
| IC selection | variance penalty / threshold cap | +0.5 × variance fraction / 0.95 |
| IC selection | pattern bonus (EOG, muscle, channel noise) | +0.30, +0.30, +0.60 |
| IC selection | variance budget (flag only) | 0.90 |
| ERP epochs | tmin / tmax / baseline / coarse reject | −0.2 s / 1.0 s / (−0.2, 0) / 500 µV p-p |
| State epochs | window / mini-epoch / overlap | 10 s / 3 s / 2 s |
| Sleep epochs | window / rejection | 20 s / none |
| AutoReject | cv / thresh method / n_interpolate / consensus / seed | 10 / bayesian_optimization / [1,2,4,8,16] / [0.5–1.0] / 42 |
| QA thresholds | bad channels / bad epochs / annotation drops | 0.20 / 0.35 / 0.50 |

---

## Discrepancies to resolve before submission (not paper text)

These were found while verifying the text above against the code and the
derivatives. They change what the Methods and the participant-flow statement can
honestly claim.

**1. The QA-exclusion statement is not currently true of every analysis.**
`exclude_failed_qa: true` is honoured by `ERPs_new/`, `Statistics/` and
`Statistics_connectivity/`, but `Stats_andrillon/config_andrillon.yaml` declares
the key while `andrillon_pipeline.py` contains no code that reads it — the CBPT
analyses (Paper 2, Section 2) currently include the QA-failed runs. Either
implement the filter or state the difference explicitly.

**2. How many runs are actually excluded is ambiguous.** `qa_summary.csv`
accumulates one row per pipeline execution rather than being rewritten: it holds
2,073 rows for 497 unique subject × run × epoch-type combinations, and 53 of
those combinations carry contradictory pass/fail verdicts from different
executions. `get_qa_exclusions()` treats *any* failing row as an exclusion, so
it marks 29 runs, whereas the most recent execution fails only 16. The file
should be deduplicated (keep last) before any number goes into the paper.

**3. QA exclusion silently never fires for subjects 02–09.** Subject IDs are
written unpadded in `qa_summary.csv` (`3`, not `03`), while the downstream
comparison uses zero-padded IDs, so `("3", "Sart1") != ("03", "Sart1")`. Of the
29 runs marked in point 2, 14 carry one-digit IDs (subjects 3, 5, 7, 8, 9) and
are therefore never excluded by anything. Combined with point 2, the analyses
that do honour QA currently drop **15 runs — a set that overlaps but does not
coincide with the 16 that fail on the latest execution** (three of those 16 have
one-digit IDs). Whatever the participant-flow statement ends up saying, it
cannot currently be derived from the code as it stands.

**4. `config.yaml` has a typo that disables one fail criterion.** Line 205 reads
`"max_bad_epochs_ratio",Holahola4`, which YAML parses as a single mangled entry
`'Holahola4\n"max_annot_dropped_epochs_ratio"'`. The
`max_annot_dropped_epochs_ratio` criterion is consequently never a cause of
failure, contrary to what the Methods above state.

**5. The line-noise QA metric is uninformative as implemented.** It compares
band power at 48–52 Hz before versus after ICA, but both recordings are already
notch-filtered, so the improvement is 0.00 dB in every run and the
`min_line_noise_db_improvement` flag fires universally. It is not among the fail
criteria, so nothing was wrongly excluded, but it should not be listed as a
quality-control criterion in the paper.

**6. The amount of variance removed by ICA is high and needs a decision.**
A median of 62.8 % of total variance is removed, with six runs above the 90 %
budget. (MNE's per-component explained-variance ratios are computed on
non-orthogonal components and are renormalised to sum to 1 in this
implementation, so the figure is an approximation — but even discounted it is
large.) Either justify it explicitly in the paper, or tighten
`iclabel_prob_min` / `variance_penalty_factor` and reprocess. This is the single
most likely reviewer objection to the pipeline.

**7. Sleep epochs receive no amplitude-based rejection at all.** They are built
with `reject=None` and `reject_by_annotation=False` and skip AutoReject; the
code comment claiming "fixed reject thresholds only" describes something that is
not passed. If sleep-window markers are reported, say plainly that those epochs
are ICA-cleaned only.

**8. `DISABLE_ICLABEL=1` in the SLURM script is a dead flag.** No Python code
reads it, and `qa_summary.csv` contains ICLabel class names
(`eye blink[28%]`, `muscle artifact[5%]`, …), confirming ICLabel ran for every
run. `Preprocessing_pipeline_new/README.md` claims the opposite and should be
corrected.

**9. Minor.** `sr_target: 250` in the config is read but never used (resampling
was moved to harmonisation) and should be removed or commented as such; the
montage is loaded from a path relative to the working directory
(`"Preprocessing_pipeline_new/CACS-64_REF.bvef"`), which breaks the
no-hardcoded-paths rule; and several `try/except` blocks in
`preprocessing_pipeline.py` swallow failures in reporting and epoching — the
batch loop catching all exceptions is the reason sub-12/`Sart4` and
sub-38/`Sart4` left partial outputs behind rather than failing loudly.
