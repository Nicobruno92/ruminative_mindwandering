"""Compute per-subject, per-channel PTP thresholds for slow-wave detection.

This script implements a threshold strategy adapted from Andrillon et al.
2021 / Pinggal et al. 2022 for a study WITHOUT a pharmacologically-defined
reference condition (no placebo arm).

Design
------
For each subject:

1. Locate every ``desc-{epoch_desc}_epo.fif`` for the configured task list
   (Sart1..Sart4) under ``derivatives_dir/sub-XX/eeg/``. There is exactly
   one session per subject and no ``ses-XX`` segment in the path.
2. Re-reference to the configured mastoid pair (matches Junifer's marker).
3. Detect all slow-wave candidates using the *exact* detection backend that
   ``SlowWavesDetection`` will call at inference time
   (``SlowWavesDetectionBase._detect_with_custom_method``).
4. Apply the same pre-percentile filters the marker applies (frequency cap,
   positive-peak artifact cap, max-PTP cap, ± proximity_window proximity
   rule), so the percentile is computed on the same wave population the
   marker will see in step 1.
5. Compute the per-channel PTP percentile across the pooled, post-filter
   wave pool from ALL four blocks combined.
6. Write a single CSV consumed by ``SlowWavesDetection`` via
   ``ptp_thresholds_path`` (columns: subject, channel, ptp_threshold,
   n_waves_used, percentile).

Pooling all four blocks keeps the threshold *condition-blind by
construction*: it does not preferentially sample on-task or
mind-wandering probes, so the within-subject contrast between those
conditions on slow-wave density (computed in step 1 with this fixed
threshold) remains valid (Bernardi 2015; Hung 2013; Vyazovskiy 2011).

The script is deterministic and side-effect free except for the output CSV.

Usage
-----
::

    python compute_thresholds.py --config config.yaml [--subject sub-04]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import pandas as pd
import yaml

import numpy as np

from junifer_eeg.markers.slow_waves_detection._slow_waves_detection_base import (
    SlowWavesDetectionBase,
)


# =============================================================================
# Config / IO helpers
# =============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Resolve project root via ``git rev-parse --show-toplevel``.

    Returns
    -------
    Path
        Absolute path of the repository root.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def resolve_path(project_root: Path, value: str) -> Path:
    """Return ``value`` as absolute path, anchored at ``project_root`` if relative.

    Parameters
    ----------
    project_root : Path
        Repo root resolved via ``get_project_root``.
    value : str
        Path string from the YAML config.

    Returns
    -------
    Path
        Absolute path.
    """
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def find_subject_fif_paths(
    derivatives_dir: Path,
    subject: str,
    tasks: List[str],
    epoch_desc: str,
) -> List[Path]:
    """Locate all ``desc-{epoch_desc}`` .fif files for one subject across tasks.

    The data layout has no ``ses-XX`` segment; files live directly under
    ``derivatives_dir/{subject}/eeg/``.

    Parameters
    ----------
    derivatives_dir : Path
        BIDS derivatives root.
    subject : str
        Subject id (``sub-XX``).
    tasks : list of str
        Task names (e.g. ``["Sart1", "Sart2", "Sart3", "Sart4"]``).
    epoch_desc : str
        BIDS ``desc-`` value identifying the pre-probe window epochs.

    Returns
    -------
    list of Path
        Existing .fif files, one per task that has data.
    """
    eeg_dir = derivatives_dir / subject / "eeg"
    paths: List[Path] = []
    for task in tasks:
        fif = eeg_dir / f"{subject}_task-{task}_desc-{epoch_desc}_epo.fif"
        if fif.exists():
            paths.append(fif)
    return paths


def discover_subjects(
    derivatives_dir: Path,
    tasks: List[str],
    epoch_desc: str,
) -> List[str]:
    """Discover all subjects with at least one matching .fif under derivatives.

    Parameters
    ----------
    derivatives_dir : Path
        BIDS derivatives root.
    tasks : list of str
        Task names that count as a valid subject.
    epoch_desc : str
        BIDS desc identifying the epoch type.

    Returns
    -------
    list of str
        Sorted list of subject ids (``sub-XX``).
    """
    subjects: List[str] = []
    for sub_dir in sorted(derivatives_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        if find_subject_fif_paths(derivatives_dir, sub_dir.name, tasks, epoch_desc):
            subjects.append(sub_dir.name)
    return subjects


# =============================================================================
# Detection (delegates to junifer_eeg)
# =============================================================================

def re_reference(
    epochs: mne.BaseEpochs, reference_channels: Tuple[str, ...]
) -> mne.BaseEpochs:
    """Re-reference epochs to mastoid channels matching Junifer's marker.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Loaded epochs.
    reference_channels : tuple of str
        Mastoid pair used as reference.

    Returns
    -------
    mne.BaseEpochs
        Re-referenced copy of ``epochs``.
    """
    missing = [ch for ch in reference_channels if ch not in epochs.ch_names]
    if missing:
        raise ValueError(
            f"Reference channels not found in epochs: {missing}. "
            f"Available: {epochs.ch_names[:5]}..."
        )
    epochs_copy = epochs.copy()
    ref_data = epochs_copy.get_data(picks=list(reference_channels)).mean(axis=1)
    ref_data = ref_data[:, np.newaxis, :]
    epochs_copy._data -= ref_data
    return epochs_copy


def detect_waves_in_epochs(
    epochs: mne.BaseEpochs,
    detection_method: str,
    freq_sw: Tuple[float, float],
    amp_ptp_initial: float,
    filter_design: str = "chebyshev2",
) -> Tuple[pd.DataFrame, Dict[int, "np.ndarray"], float]:
    """Detect raw slow-wave events in one Epochs object.

    Returns
    -------
    wave_df : pd.DataFrame
        Long-format DataFrame (one row per detected wave) with the columns
        produced by ``SlowWavesDetectionBase`` for the selected backend.
    filtered_per_epoch : dict[int, np.ndarray]
        Bandpass-filtered, re-referenced signal in µV per epoch index.
        Required to apply the proximity-to-artifact rule downstream.
    sf : float
        Sampling frequency of the (possibly post-resampled) signal.
    """
    sw_base = SlowWavesDetectionBase()
    sf = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    chan2idx = {ch: i for i, ch in enumerate(ch_names)}

    if detection_method == "custom":
        event_dfs, filtered_per_epoch = sw_base._detect_with_custom_method(
            epochs_copy=epochs,
            sf=sf,
            ch_names=ch_names,
            chan2idx=chan2idx,
            freq_sw=freq_sw,
            amp_ptp_initial=amp_ptp_initial,
            filter_design=filter_design,
        )
    elif detection_method == "yasa":
        event_dfs, filtered_per_epoch = sw_base._detect_with_yasa(
            epochs_copy=epochs,
            sf=sf,
            ch_names=ch_names,
            chan2idx=chan2idx,
            freq_sw=freq_sw,
            amp_ptp_initial=amp_ptp_initial,
            need_filtered=True,
        )
    else:
        raise ValueError(
            f"Unknown detection_method={detection_method!r}; "
            "expected 'custom' or 'yasa'"
        )

    if not event_dfs:
        return pd.DataFrame(), filtered_per_epoch, sf
    return pd.concat(event_dfs, ignore_index=True), filtered_per_epoch, sf


def apply_basic_filters(
    sw_df: pd.DataFrame,
    freq_threshold: float,
    artifact_threshold: float,
    max_ptp_amplitude: float,
) -> pd.DataFrame:
    """Apply the same pre-percentile filters as Junifer's marker.

    Mirrors the logic in ``_apply_dynamic_threshold`` so the threshold pool
    is computed on exactly the wave population the marker will see at
    inference time (modulo the percentile step itself).

    Parameters
    ----------
    sw_df : pd.DataFrame
        Detected wave events.
    freq_threshold : float
        Maximum allowed wave frequency (Hz).
    artifact_threshold : float
        Maximum allowed positive-peak amplitude (µV).
    max_ptp_amplitude : float
        Maximum allowed peak-to-peak amplitude (µV).

    Returns
    -------
    pd.DataFrame
        Filtered events.
    """
    if sw_df.empty:
        return sw_df

    sw_df = sw_df[sw_df["Frequency"] <= freq_threshold].copy()
    if sw_df.empty:
        return sw_df

    sw_df = sw_df[sw_df["PTP"] < max_ptp_amplitude].copy()
    if sw_df.empty:
        return sw_df

    pos_peak_col: Optional[str] = None
    if "ValPosPeak" in sw_df.columns:
        pos_peak_col = "ValPosPeak"
    elif "PosPeak" in sw_df.columns:
        pos_peak_col = "PosPeak"
    if pos_peak_col is not None:
        sw_df = sw_df[sw_df[pos_peak_col] < artifact_threshold].copy()

    return sw_df


def compute_subject_thresholds(
    fif_paths: List[Path],
    detection: Dict[str, Any],
    filters: Dict[str, Any],
    percentile: float,
) -> pd.DataFrame:
    """Run detection across all .fif files for one subject and compute thresholds.

    Pools waves across the supplied .fif files (all four task blocks) and
    returns one row per channel with the percentile of PTP.

    Parameters
    ----------
    fif_paths : list of Path
        Pre-probe-window epoch files for the subject.
    detection : dict
        ``method``, ``freq_sw``, ``amp_ptp_initial``, ``reference_channels``,
        and optional ``filter_design`` (default ``chebyshev2``).
    filters : dict
        ``freq_threshold``, ``artifact_threshold``, ``max_ptp_amplitude``,
        and optional ``proximity_amplitude`` / ``proximity_window``.
    percentile : float
        Percentile (0-100) of PTP defining the per-channel cutoff.

    Returns
    -------
    pd.DataFrame
        Columns ``channel``, ``ptp_threshold``, ``n_waves_used``,
        ``freq_p50_post_filter``. The frequency median is QC-only — it does
        not feed the marker; downstream stats can use it to flag subjects
        whose threshold pool is dominated by theta-band events
        (``freq_p50_post_filter > 4 Hz`` ⇒ candidate for sensitivity
        analysis).
    """
    reference_channels = tuple(detection["reference_channels"])
    freq_sw = tuple(detection["freq_sw"])
    amp_ptp_initial = float(detection["amp_ptp_initial"])
    detection_method = detection["method"]
    filter_design = detection.get("filter_design", "chebyshev2")

    proximity_amplitude = filters.get("proximity_amplitude")
    proximity_window = float(filters.get("proximity_window", 1.0))

    sw_base = SlowWavesDetectionBase()

    all_waves: List[pd.DataFrame] = []
    for fif_path in fif_paths:
        print(f"  Loading {fif_path.name}")
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        # Match Junifer's filter_to_eeg_channels behaviour.
        epochs = epochs.pick_types(eeg=True)
        epochs_ref = re_reference(epochs, reference_channels)
        wave_df, filtered_per_epoch, sf = detect_waves_in_epochs(
            epochs=epochs_ref,
            detection_method=detection_method,
            freq_sw=freq_sw,
            amp_ptp_initial=amp_ptp_initial,
            filter_design=filter_design,
        )
        n_raw = len(wave_df)

        wave_df = apply_basic_filters(
            wave_df,
            freq_threshold=float(filters["freq_threshold"]),
            artifact_threshold=float(filters["artifact_threshold"]),
            max_ptp_amplitude=float(filters["max_ptp_amplitude"]),
        )

        if proximity_amplitude is not None and not wave_df.empty:
            n_pre_prox = len(wave_df)
            wave_df = sw_base._apply_proximity_filter(
                sw_df=wave_df,
                filtered_per_epoch=filtered_per_epoch,
                sf=sf,
                proximity_amplitude=float(proximity_amplitude),
                proximity_window=proximity_window,
            )
            n_post_prox = len(wave_df)
            print(
                f"    detected {n_raw} raw → {n_pre_prox} after basic filters "
                f"→ {n_post_prox} after proximity (±{proximity_window}s, "
                f"{proximity_amplitude} µV)"
            )
        else:
            print(
                f"    detected {n_raw} raw → {len(wave_df)} after basic filters"
            )

        wave_df["source_file"] = fif_path.name
        all_waves.append(wave_df)

    pooled = (
        pd.concat(all_waves, ignore_index=True) if all_waves else pd.DataFrame()
    )
    if pooled.empty:
        raise RuntimeError(
            f"No waves survived filtering across files: "
            f"{[p.name for p in fif_paths]}"
        )

    print(f"  pooled (post all filters): {len(pooled)} waves")

    grouped = pooled.groupby("Channel")
    thr = grouped["PTP"].quantile(percentile / 100.0).rename("ptp_threshold")
    counts = grouped.size().rename("n_waves_used")
    freq_p50 = grouped["Frequency"].median().rename("freq_p50_post_filter")
    return (
        pd.concat([thr, counts, freq_p50], axis=1)
        .reset_index()
        .rename(columns={"Channel": "channel"})
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Entry point — parse args, run threshold computation, write CSV."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-subject, per-channel PTP thresholds for slow-wave "
            "detection by pooling all four task blocks of the subject's "
            "single session."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process only this subject (e.g., sub-04). If omitted, all available.",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    cfg = load_config(Path(args.config))

    project = cfg["project"]
    derivatives_dir = resolve_path(project_root, project["derivatives_dir"])
    output_csv = resolve_path(project_root, project["output_csv"])
    tasks: List[str] = list(cfg["tasks"])
    epoch_desc: str = cfg["epoch_desc"]
    detection: Dict[str, Any] = cfg["detection"]
    filters: Dict[str, Any] = cfg["filters"]
    percentile: float = float(cfg["percentile"])
    cfg_subjects = cfg.get("subjects")

    if args.subject is not None:
        subjects_to_run = [args.subject]
    elif cfg_subjects:
        subjects_to_run = list(cfg_subjects)
    else:
        subjects_to_run = discover_subjects(derivatives_dir, tasks, epoch_desc)

    if not subjects_to_run:
        raise RuntimeError(
            f"No subjects to process. derivatives_dir={derivatives_dir!s}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    skipped: List[str] = []

    for subject in subjects_to_run:
        print(f"\n=== {subject} ===")
        fif_paths = find_subject_fif_paths(
            derivatives_dir=derivatives_dir,
            subject=subject,
            tasks=tasks,
            epoch_desc=epoch_desc,
        )
        if not fif_paths:
            print(
                f"  [SKIP] No desc-{epoch_desc} .fif files found for "
                f"{subject} (tasks={tasks})"
            )
            skipped.append(subject)
            continue

        print(f"  Found {len(fif_paths)} block(s): "
              f"{[p.name for p in fif_paths]}")

        thr_df = compute_subject_thresholds(
            fif_paths=fif_paths,
            detection=detection,
            filters=filters,
            percentile=percentile,
        )
        thr_df.insert(0, "subject", subject)
        thr_df["percentile"] = percentile
        rows.append(thr_df)
        print(
            f"  → {len(thr_df)} channel thresholds "
            f"(median PTP threshold = {thr_df['ptp_threshold'].median():.2f} µV, "
            f"median n_waves/channel = "
            f"{int(thr_df['n_waves_used'].median())})"
        )

    if not rows:
        raise RuntimeError("No subjects processed; nothing to write.")

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(
        f"Wrote {len(out_df)} rows for {out_df['subject'].nunique()} subjects"
    )
    print(f"Output: {output_csv}")
    if skipped:
        print(f"Skipped (no desc-{epoch_desc} data): {skipped}")


if __name__ == "__main__":
    main()
