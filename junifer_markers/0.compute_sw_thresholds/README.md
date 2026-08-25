# Step 0 — Per-subject PTP thresholds for slow-wave detection

Computes the per-subject, per-channel **90th-percentile peak-to-peak amplitude
threshold** that step 1 (Junifer `SlowWavesDetection`) will use as a fixed
cutoff for "large-amplitude" slow waves.

## Why pooled-4-blocks (and not placebo)

Andrillon 2021 / Pinggal 2022 derive the threshold from a placebo session
that is *not* one of the experimental contrasts. We have no placebo arm — the
contrast of interest is **on-task vs mind-wandering** within the same session.

The threshold pool **must not coincide with either condition being contrasted**,
otherwise the detection cutoff is calibrated on (and biased toward) one of the
arms:

* **Pool = on-task probes** → MW probes get fewer waves above threshold simply
  because the threshold tracks on-task amplitude statistics.
* **Pool = MW probes** → symmetric problem in the opposite direction.

Pooling **all four task blocks** keeps the threshold *condition-blind by
construction*: it reflects each subject's global EEG calibration, not the
content of a sub-set, while maximising N waves for stable percentile
estimation (Bernardi 2015; Hung 2013; Vyazovskiy 2011).

> *"Slow-wave detection followed the established pipeline (Andrillon et al.
> 2021; Pinggal et al. 2022). In the absence of a pharmacologically-defined
> reference condition, the per-channel 90th-percentile peak-to-peak amplitude
> threshold was computed for each subject from the pooled distribution of
> waves detected across all four task blocks, and applied as a fixed cutoff
> to compute slow-wave density per pre-probe window. This preserves
> within-subject comparability of slow-wave density between on-task and
> mind-wandering probes while remaining condition-blind by construction
> (Bernardi et al. 2015; Hung et al. 2013; Vyazovskiy et al. 2011)."*

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Detection/filter parameters (must match step 1 `config_sleep.yaml`) |
| `compute_thresholds.py` | Pools waves across `Sart1..Sart4` and writes the threshold CSV |
| `slurm_compute_thresholds.sh` | SLURM wrapper |

## Pipeline (per subject)

1. Locate every `desc-sleep_epo.fif` for `Sart1..Sart4` under
   `derivatives/sub-XX/eeg/` (single session, no `ses-XX` segment).
2. Re-reference to `TP9, TP10` (matches `SlowWavesDetection` in step 1).
3. Run the **same detection backend** as Junifer's marker
   (`SlowWavesDetectionBase._detect_with_custom_method`, imported directly).
4. Apply the same pre-percentile filters as the marker:
   - `Frequency ≤ 7 Hz`
   - `PosPeak < 75 µV`
   - `PTP < 150 µV`
   - proximity rule (drop waves within ±1 s of any sample exceeding 150 µV).
5. Compute the per-channel 90th-percentile PTP across the **pooled** wave
   set from all four blocks.
6. Write `BIDS/features/sw_thresholds/pooled_sw_thresholds.csv`:

   ```csv
   subject,channel,ptp_threshold,n_waves_used,freq_p50_post_filter,percentile
   sub-04,Fp1,42.7,113,5.10,90.0
   sub-04,Fp2,41.3,108,5.07,90.0
   ...
   ```

   `subject` is the BIDS id (`sub-XX`); step 1 looks the row up by exact
   match between the element subject and this column.

   Columns:

   | Column | Use |
   |---|---|
   | `subject` | BIDS id; matched verbatim against the Junifer element subject |
   | `channel` | EEG channel name |
   | `ptp_threshold` | The 90th-percentile PTP cutoff that step 1 will apply |
   | `n_waves_used` | Pool size used to estimate the percentile (≥200 ⇒ stable) |
   | `freq_p50_post_filter` | Median Hz of the threshold pool — QC only |
   | `percentile` | The percentile used (constant per run; here 90) |

   `freq_p50_post_filter` is **QC-only** (not consumed by the marker). Use it
   to flag subjects whose pool is dominated by theta-band events:
   `freq_p50_post_filter > 4 Hz` ⇒ candidate for sensitivity analysis (true
   slow waves cluster in 1-3 Hz; theta-dominant subjects may be measuring
   high-amplitude theta, not delta/SO).

## Running

Local smoke test, single subject:

```bash
cd junifer_markers/0.compute_sw_thresholds
python compute_thresholds.py --config config.yaml --subject sub-04
```

All discovered subjects on the cluster:

```bash
sbatch slurm_compute_thresholds.sh
```

Single subject on the cluster:

```bash
SUBJECT=sub-04 sbatch slurm_compute_thresholds.sh
```

## Coupling with step 1

`config.yaml` here and the `SlowWavesDetection` entries in
`1.markers_h5_creation/config_sleep.yaml` MUST share the same detection /
filter parameters. Any drift means the pool used for the percentile differs
from the pool the marker filters at inference time, breaking the semantics.

Mandatory shared parameters:

| Parameter | Value |
|-----------|-------|
| `detection_method` | `"custom"` |
| `freq_sw` | `[1.0, 10.0]` |
| `amp_ptp_initial` | `15.0` |
| `freq_threshold` | `7.0` |
| `artifact_threshold` | `75.0` |
| `max_ptp_amplitude` | `150.0` |
| `proximity_amplitude` | `150.0` |
| `proximity_window` | `1.0` |
| `reference_channels` | `["TP9", "TP10"]` |
| `ptp_percentile` | `90.0` |

Step 1 should set `ptp_thresholds_strict: true`: if a subject is missing from
the CSV, the run must fail loudly rather than silently fall back to an
in-element percentile.
