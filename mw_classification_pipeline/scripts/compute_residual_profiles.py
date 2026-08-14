#!/usr/bin/env python
"""
Marker |SHAP| profiles for each dimension, full vs residualized.

Feeds the "does residualizing change which features matter?" figure in
``generate_combined_classification_figure.py``. Kept as a separate step because
building it means reading 100 stacked SHAP pickles per contrast per pipeline —
28 combinations, tens of minutes — and the figure script is run often and should
stay fast. The figure reads the table this writes and skips itself with a
message if it is absent.

Each dimension is paired with the contrast that residualizes it against the
other probe dimensions (within-subject OLS). Comparing the two answers whether a
dimension's decodable signal is its own or is inherited from what it correlates
with: if the same markers carry it before and after, the signal is specific; if
the profile reorders, it was shared.

Profiles are written on their **absolute** mean(|SHAP|) scale. Shares are
included for reference but are relative — a marker's share falls whenever the
others rise — so comparisons between two contrasts must use the absolute value
or the marker ranking, never the share.

USAGE
-----
    conda activate ML
    python mw_classification_pipeline/scripts/compute_residual_profiles.py
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Configuration
# =============================================================================
_SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = _SCRIPTS_DIR.parent
RESULTS_ROOT = PIPELINE_ROOT / "results" / "MW_Classification"
OUT_PATH = PIPELINE_ROOT / "results" / "feature_consistency" / "residual_marker_profiles.csv"
CONFIG_PATH = _SCRIPTS_DIR / "config_feature_consistency.yaml"

EXPECTED_N_FEATURES = 175
EXPECTED_N_MARKERS = 23

# (dimension key, label, full dir, residualized dir) per pipeline.
# The full on/off directory differs in case between pipelines, a
# historical naming inconsistency also carried by config_feature_consistency.yaml.
CONTRAST_PAIRS = [
    {"key": "on_off", "label": "On/Off-Task",
     "ws": ("on_vs_off_within_median", "onoff_within_median_res"),
     "loso": ("ON_vs_OFF_within_median", "onoff_within_median_res")},
    {"key": "valence", "label": "Valence",
     "ws": ("valence_within_median", "valence_within_median_res"),
     "loso": ("valence_within_median", "valence_within_median_res")},
    {"key": "confidence", "label": "Confidence",
     "ws": ("confidence_within_median", "confidence_within_median_res"),
     "loso": ("confidence_within_median", "confidence_within_median_res")},
    {"key": "selfother", "label": "Self/Other",
     "ws": ("selfother_within_median", "selfother_within_median_res"),
     "loso": ("selfother_within_median", "selfother_within_median_res")},
    {"key": "time", "label": "Time",
     "ws": ("time_within_median", "time_within_median_res"),
     "loso": ("time_within_median", "time_within_median_res")},
    # valence_sq / time_sq (paired with the *_res_cross residualized contrasts)
    # removed 2026-08-13 together with the quadratic construct itself — see
    # CLAUDE.md "Quadratic Terms: Removed". Their run directories still exist
    # under results/MW_Classification/, so this list is what keeps them out of
    # the profiles table, not the absence of data on disk.
]

PIPELINE_SPEC = {
    "ws": ("WithinSubject", "*_shap_values_stacked.pkl"),
    "loso": ("LOSO", "rf_loso_*_shap_values.pkl"),
}

# =============================================================================
# Loading
# =============================================================================


def _run_index(run_dir: Path) -> int:
    """Extract the integer run index from a ``run_N`` directory name."""
    match = re.fullmatch(r"run_(\d+)", run_dir.name)
    if match is None:
        raise ValueError(f"Unexpected run directory name: {run_dir}")
    return int(match.group(1))


def mean_abs_shap_profile(base_dir: Path, shap_glob: str) -> tuple[np.ndarray, list[str]]:
    """
    Mean |SHAP| per feature, over all trials and all runs of one contrast.

    No per-subject segmentation: this figure compares group-level profiles
    between two labellings of the same data, so the quantity of interest is the
    model's overall attribution.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(profile, feature_names)`` with ``profile`` of shape (n_features,).
    """
    run_dirs = sorted(base_dir.glob("true_runs/run_*"), key=_run_index)
    if not run_dirs:
        raise FileNotFoundError(f"No true_runs/run_* directories under {base_dir}")

    per_run: list[np.ndarray] = []
    feature_names: list[str] = []
    for run_dir in run_dirs:
        paths = sorted(run_dir.glob(shap_glob))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one SHAP pickle matching '{shap_glob}' in "
                f"{run_dir}; found {len(paths)}."
            )
        with open(paths[0], "rb") as handle:
            payload = pickle.load(handle)
        shap_values = np.asarray(payload["shap_values"])
        run_features = [str(f) for f in payload["feature_names"]]

        if shap_values.shape[1] != EXPECTED_N_FEATURES:
            raise ValueError(
                f"{run_dir}: {shap_values.shape[1]} features, expected "
                f"{EXPECTED_N_FEATURES}. A mismatch means this run used a "
                f"different feature space and the comparison would be invalid."
            )
        if not feature_names:
            feature_names = run_features
        elif run_features != feature_names:
            raise ValueError(f"{run_dir}: feature name/order differs from run 0.")

        per_run.append(np.abs(shap_values).mean(axis=0))

    return np.mean(per_run, axis=0), feature_names


def build_marker_map(feature_names: list[str], marker_regex: str) -> tuple[list[str], np.ndarray]:
    """Collapse feature columns onto the 23 CBPT markers; see the analysis script."""
    pattern = re.compile(marker_regex)
    markers = []
    for name in feature_names:
        stripped = pattern.sub("", name)
        if stripped == name:
            raise ValueError(
                f"Feature '{name}' does not match the marker grammar "
                f"'{marker_regex}'; the naming convention changed."
            )
        markers.append(stripped)
    unique = sorted(set(markers))
    if len(unique) != EXPECTED_N_MARKERS:
        raise ValueError(f"Derived {len(unique)} markers, expected {EXPECTED_N_MARKERS}.")
    index = {name: i for i, name in enumerate(unique)}
    return unique, np.array([index[name] for name in markers])


def to_marker_profile(profile: np.ndarray, marker_index: np.ndarray, n_markers: int) -> np.ndarray:
    """
    Per-ROI mean |SHAP| per marker, on its original absolute scale.

    Mean and not sum: the four evoked markers own one column each and the
    nineteen sleep markers own nine, so summing would hand the sleep markers up
    to nine times the mass before any data is consulted.

    **Not** normalised to sum to 1. A share is relative — a marker's share falls
    whenever the others rise, so "this marker dropped after residualizing" can
    be true of the share while the absolute attribution is unchanged, and a
    figure built on shares cannot tell the two apart. The share is still written
    alongside for reference, but every comparison should use the absolute value.
    """
    totals = np.zeros(n_markers)
    np.add.at(totals, marker_index, profile)
    return totals / np.bincount(marker_index, minlength=n_markers)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Write one row per (dimension, pipeline, marker) with both profiles."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    marker_regex = config["marker_regex"]

    rows = []
    for pair in CONTRAST_PAIRS:
        for pipeline, (family_dir, shap_glob) in PIPELINE_SPEC.items():
            full_dir, residual_dir = pair[pipeline]
            profiles = {}
            for kind, directory in (("full", full_dir), ("residual", residual_dir)):
                base = RESULTS_ROOT / family_dir / directory / "all" / "rf"
                print(f"  {pair['key']:<12} {pipeline:<5} {kind:<9} {directory}", flush=True)
                profile, feature_names = mean_abs_shap_profile(base, shap_glob)
                markers, marker_index = build_marker_map(feature_names, marker_regex)
                profiles[kind] = to_marker_profile(profile, marker_index, len(markers))

            totals = {kind: float(values.sum()) for kind, values in profiles.items()}
            for i, marker in enumerate(markers):
                rows.append({
                    "contrast": pair["key"], "label": pair["label"],
                    "pipeline": pipeline, "marker": marker,
                    "abs_full": float(profiles["full"][i]),
                    "abs_residual": float(profiles["residual"][i]),
                    "share_full": float(profiles["full"][i] / totals["full"]),
                    "share_residual": float(profiles["residual"][i] / totals["residual"]),
                    "dir_full": full_dir, "dir_residual": residual_dir,
                })

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(frame)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
