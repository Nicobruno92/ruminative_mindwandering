"""
Reader module for connectivity statistics pipeline.

Loads wSMI connectivity pair data from aggregated CSV files and optionally
aggregates channel-pair values into ROI-pair values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations_with_replacement


# =============================================================================
# CONSTANTS
# =============================================================================

# Separator used in channel pair column names.
# The Junifer H5 reader writes pairs as "Ch1-Ch2" (single hyphen).
PAIR_SEPARATOR = "-"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_connectivity_data(
    features_root: str,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    epoch_types: Optional[List[str]] = None,
    bands: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all connectivity CSV files into a long-format DataFrame.

    Scans for files matching `*_connectivity.csv` pattern and extracts
    wSMI pair values for each probe.

    Parameters
    ----------
    features_root : str
        Root directory containing features/sub-XX/eeg/junifer_aggregated/
    subjects : Optional[List[str]]
        Subject IDs to include (e.g., ["02", "03"]). None = all.
    tasks : Optional[List[str]]
        Tasks to include (e.g., ["Sart1"]). None = all.
    epoch_types : Optional[List[str]]
        Epoch types to include (e.g., ["state_connectivity"]). None = all.
    bands : Optional[List[str]]
        Frequency bands to include (e.g., ["theta", "alpha"]). None = all.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        subject, task, probe_number, label, n_epochs, onoff,
        epoch_type, band, connection_id, wsmi_value
    """
    features_path = Path(features_root)
    assert features_path.exists(), f"Features root not found: {features_root}"

    # Find all connectivity CSV files
    # Use rglob for reliable recursive search (handles symlinks better than glob with **)
    csv_files = sorted(features_path.rglob("*_connectivity.csv"))
    # Filter to only files in junifer_aggregated directories
    csv_files = [f for f in csv_files if "junifer_aggregated" in str(f)]

    if verbose:
        print(f"Found {len(csv_files)} connectivity CSV files")

    assert len(csv_files) > 0, (
        f"No connectivity CSV files found in {features_root}"
    )

    # Filter files
    filtered_files = _filter_files(csv_files, subjects, tasks, epoch_types)

    if verbose:
        print(f"After filtering: {len(filtered_files)} files to load")

    # Load and combine – each file returns a DataFrame (melt-based, no iterrows)
    all_dfs = []
    for file_path in filtered_files:
        file_df = _parse_connectivity_csv(file_path, bands)
        if not file_df.empty:
            all_dfs.append(file_df)

    assert len(all_dfs) > 0, (
        f"All connectivity CSV files were empty after parsing in {features_root}"
    )

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"  Loaded {len(combined_df)} connection records")

    if verbose:
        n_subjects = combined_df["subject"].nunique()
        n_probes = len(
            combined_df.groupby(["subject", "task", "probe_number"]).ngroup()
            .unique()
        )
        n_connections = combined_df["connection_id"].nunique()
        print(f"  Subjects: {n_subjects}")
        print(f"  Probes: {n_probes}")
        print(f"  Unique connections: {n_connections}")

    return combined_df


def _filter_files(
    csv_files: List[Path],
    subjects: Optional[List[str]],
    tasks: Optional[List[str]],
    epoch_types: Optional[List[str]],
) -> List[Path]:
    """
    Filter CSV files by subject, task, and epoch type.

    Parameters
    ----------
    csv_files : List[Path]
        All discovered CSV files.
    subjects : Optional[List[str]]
        Subject filter.
    tasks : Optional[List[str]]
        Task filter.
    epoch_types : Optional[List[str]]
        Epoch type filter.

    Returns
    -------
    List[Path]
        Filtered file paths.
    """
    filtered = []

    for fpath in csv_files:
        fname = fpath.name

        # Parse subject from filename: sub-XX_...
        subject = fname.split("_")[0].replace("sub-", "")
        if subjects is not None and subject not in subjects:
            continue

        # Parse task: ..._task-YY_...
        if "_task-" not in fname:
            continue
        task = fname.split("_task-")[1].split("_")[0]
        if tasks is not None and task not in tasks:
            continue

        # Parse epoch type from filename: ..._probe-NNN_label_TYPE_connectivity.csv
        # The epoch_type is the part before "_connectivity"
        if "_connectivity.csv" not in fname:
            continue
        # Extract the type prefix: e.g., "state" from "state_connectivity"
        parts = fname.replace(".csv", "").split("_")
        # Find parts that combine to form something like "state_connectivity"
        epoch_type = None
        for i, p in enumerate(parts):
            if p == "connectivity" and i > 0:
                epoch_type = parts[i - 1] + "_connectivity"
                break

        if epoch_types is not None and epoch_type not in epoch_types:
            continue

        filtered.append(fpath)

    return filtered


def _parse_connectivity_csv(
    file_path: Path,
    bands: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parse a single connectivity CSV file into a long-format DataFrame.

    Uses pd.melt instead of iterrows to avoid building one Python dict per
    (pair × band) — critical for processing thousands of files without OOM.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.
    bands : Optional[List[str]]
        Bands to include. None = all.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: subject, task, probe_number, label,
        n_epochs, onoff, valence, selfother, confidence, time,
        epoch_type, band, connection_id, wsmi_value.
        Empty DataFrame if no matching marker columns are found.
    """
    df = pd.read_csv(file_path)

    # ── Scalar metadata (constant for every row in this file) ────────────
    subject = str(df["subject"].iloc[0])
    task = str(df["task"].iloc[0])
    probe_number = int(df["probe_number"].iloc[0])
    label = str(df["label"].iloc[0])
    n_epochs = int(df["n_epochs"].iloc[0])
    # onoff: derive from ontask_label (ONTASK=100, OFFTASK=0) when not stored
    # directly. This matches the convention in the aggregation helpers.
    if "onoff" in df.columns:
        onoff_val = float(df["onoff"].iloc[0])
    elif "ontask_label" in df.columns:
        ontask = str(df["ontask_label"].iloc[0]).upper()
        onoff_val = 100.0 if ontask == "ONTASK" else 0.0 if ontask == "OFFTASK" else np.nan
    elif "label" in df.columns:
        label_str = str(df["label"].iloc[0]).lower()
        onoff_val = 100.0 if "ontask" in label_str else 0.0 if "offtask" in label_str else np.nan
    else:
        onoff_val = np.nan
    valence = float(df["valence"].iloc[0]) if "valence" in df.columns else np.nan
    selfother = float(df["selfother"].iloc[0]) if "selfother" in df.columns else np.nan
    confidence = float(df["confidence"].iloc[0]) if "confidence" in df.columns else np.nan
    time_val = float(df["time"].iloc[0]) if "time" in df.columns else np.nan

    # ── Epoch type from filename ──────────────────────────────────────────
    fname = file_path.name.replace(".csv", "")
    parts = fname.split("_")
    epoch_type = "unknown"
    for i, p in enumerate(parts):
        if p == "connectivity" and i > 0:
            epoch_type = parts[i - 1] + "_connectivity"
            break

    # ── Identify wSMI pair marker columns ────────────────────────────────
    marker_cols = [
        c for c in df.columns
        if c.startswith("wsmi_") and "pairs" in c
    ]
    if bands is not None:
        marker_cols = [
            c for c in marker_cols
            if _extract_band_from_column(c) in bands
        ]

    if not marker_cols:
        return pd.DataFrame()

    # ── Melt: (n_pairs rows × n_band_cols) → (n_pairs × n_bands rows) ────
    # Avoids per-row Python dict allocation; stays in numpy/pandas internals.
    result = (
        df[["channel"] + marker_cols]
        .melt(
            id_vars=["channel"],
            value_vars=marker_cols,
            var_name="marker_col",
            value_name="wsmi_value",
        )
        .rename(columns={"channel": "connection_id"})
    )
    result["band"] = result["marker_col"].apply(_extract_band_from_column)
    result = result.drop(columns=["marker_col"])

    # ── Attach scalar metadata as columns ────────────────────────────────
    result["subject"] = subject
    result["task"] = task
    result["probe_number"] = probe_number
    result["label"] = label
    result["n_epochs"] = n_epochs
    result["onoff"] = onoff_val
    result["valence"] = valence
    result["selfother"] = selfother
    result["confidence"] = confidence
    result["time"] = time_val
    result["epoch_type"] = epoch_type

    out_cols = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "valence", "selfother", "confidence", "time",
        "epoch_type", "band", "connection_id", "wsmi_value",
    ]
    return result[[c for c in out_cols if c in result.columns]]


def _extract_band_from_column(col_name: str) -> str:
    """
    Extract frequency band name from a wSMI column name.

    Parameters
    ----------
    col_name : str
        Column name like 'wsmi_theta_pairs_trimmean'.

    Returns
    -------
    str
        Band name (e.g., 'theta').
    """
    # Pattern: wsmi_{band}_pairs_{method}
    parts = col_name.split("_")
    # Find index of "wsmi" and return next element
    for i, p in enumerate(parts):
        if p == "wsmi" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


# =============================================================================
# ROI AGGREGATION
# =============================================================================

def build_channel_to_roi_map(rois: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build a mapping from channel name to ROI name.

    Parameters
    ----------
    rois : Dict[str, List[str]]
        ROI definitions: {roi_name: [ch1, ch2, ...]}.

    Returns
    -------
    Dict[str, str]
        Mapping: {channel_name: roi_name}.
    """
    ch_to_roi = {}
    for roi_name, channels in rois.items():
        for ch in channels:
            ch_to_roi[ch] = roi_name
    return ch_to_roi


def get_roi_pair_labels(rois: Dict[str, List[str]]) -> List[str]:
    """
    Generate sorted list of all unique ROI pair labels.

    Includes both within-ROI and between-ROI pairs.

    Parameters
    ----------
    rois : Dict[str, List[str]]
        ROI definitions.

    Returns
    -------
    List[str]
        Sorted ROI pair labels (e.g., ["frontal_left--frontal_left", ...]).
    """
    roi_names = sorted(rois.keys())
    pairs = []
    for r1, r2 in combinations_with_replacement(roi_names, 2):
        pairs.append(f"{r1}{PAIR_SEPARATOR}{r2}")
    return pairs


def aggregate_to_roi_pairs(
    df: pd.DataFrame,
    rois: Dict[str, List[str]],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate channel-pair wSMI values into ROI-pair values.

    For each probe × band, maps each channel pair to its ROI pair
    and computes the mean wSMI across all channel pairs within each ROI pair.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data (from load_connectivity_data).
    rois : Dict[str, List[str]]
        ROI definitions.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Same format as input but with ROI pair IDs instead of channel pair IDs
        in the 'connection_id' column, and values averaged.
    """
    ch_to_roi = build_channel_to_roi_map(rois)

    # Parse connection_id into channel1, channel2
    df = df.copy()
    split_cols = df["connection_id"].str.split(PAIR_SEPARATOR, n=1, expand=True)
    df["ch1"] = split_cols[0].str.strip()
    df["ch2"] = split_cols[1].str.strip()

    # Map to ROIs
    df["roi1"] = df["ch1"].map(ch_to_roi)
    df["roi2"] = df["ch2"].map(ch_to_roi)

    # Drop pairs where either channel is not in any ROI
    n_before = len(df)
    df = df.dropna(subset=["roi1", "roi2"])
    n_dropped = n_before - len(df)

    if verbose and n_dropped > 0:
        print(f"  Dropped {n_dropped} pairs with unmapped channels")

    # Normalize ROI pair ordering (alphabetical) so A--B == B--A
    roi_pair = df.apply(
        lambda r: PAIR_SEPARATOR.join(sorted([r["roi1"], r["roi2"]])),
        axis=1,
    )
    df["connection_id"] = roi_pair

    # Group by all metadata + ROI pair and average wSMI
    group_cols = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "valence", "selfother", "confidence", "time",
        "epoch_type", "band", "connection_id",
    ]
    # Keep only columns that exist
    group_cols = [c for c in group_cols if c in df.columns]

    # dropna=False: keep groups where metadata columns (valence, time, etc.)
    # are NaN — this is expected for connectivity CSVs which lack those fields.
    result = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(wsmi_value=("wsmi_value", "mean"))
    )

    if verbose:
        n_roi_pairs = result["connection_id"].nunique()
        print(f"  Aggregated to {n_roi_pairs} ROI pairs")

    return result


# =============================================================================
# DATA PREPARATION FOR LMM
# =============================================================================

def prepare_connectivity_for_lmm(
    df: pd.DataFrame,
    band: str,
    epoch_type: Optional[str] = None,
    onoff_max_value: Optional[float] = None,
    min_predictor_variability: Optional[float] = None,
    min_minority_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare connectivity data for LMM analysis for a specific band.

    Filters data, pivots to wide format (connections as columns),
    and validates the data structure.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data (ROI or channel level).
    band : str
        Frequency band to analyze (e.g., "theta").
    epoch_type : Optional[str]
        Filter to specific epoch type.
    onoff_max_value : Optional[float]
        Maximum onoff value to include.
    min_predictor_variability : Optional[float]
        Minimum within-subject range for onoff.
    min_minority_ratio : Optional[float]
        Minimum minority class proportion.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Wide-format DataFrame: rows=observations (probes), columns include
          metadata + one column per connection_id
        - List of connection_id column names
    """
    # Filter by band and epoch type
    filtered = df[df["band"] == band].copy()
    if epoch_type is not None:
        filtered = filtered[filtered["epoch_type"] == epoch_type]

    assert len(filtered) > 0, (
        f"No data for band='{band}', epoch_type='{epoch_type}'"
    )

    # Apply onoff filter
    if onoff_max_value is not None and "onoff" in filtered.columns:
        n_before = len(filtered)
        filtered = filtered[filtered["onoff"] <= onoff_max_value]
        print(f"  onoff <= {onoff_max_value}: {n_before} -> {len(filtered)} rows")

    # Pivot: rows = probes, columns = connection_ids
    # Only include metadata columns that (a) exist and (b) have at least one
    # non-NaN value — avoids pivot_table silently dropping all rows when
    # optional columns (valence, selfother, confidence, time) are absent in
    # connectivity CSVs.
    candidate_meta = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "epoch_type", "valence", "selfother", "confidence", "time",
    ]
    metadata_cols = [
        c for c in candidate_meta
        if c in filtered.columns and filtered[c].notna().any()
    ]

    pivot = filtered.pivot_table(
        index=metadata_cols,
        columns="connection_id",
        values="wsmi_value",
        aggfunc="mean",
    ).reset_index()

    connection_ids = [
        c for c in pivot.columns if c not in metadata_cols
    ]

    # Drop rows with NaN in any connection
    n_before = len(pivot)
    pivot = pivot.dropna(subset=connection_ids)
    if len(pivot) < n_before:
        print(f"  Dropped {n_before - len(pivot)} probes with NaN connections")

    # Filter subjects by predictor variability
    if min_predictor_variability is not None and "onoff" in pivot.columns:
        subject_ranges = pivot.groupby("subject")["onoff"].agg(
            lambda x: x.max() - x.min()
        )
        valid_subjects = subject_ranges[
            subject_ranges >= min_predictor_variability
        ].index
        n_excluded = pivot["subject"].nunique() - len(valid_subjects)
        pivot = pivot[pivot["subject"].isin(valid_subjects)]
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} subjects (low predictor variability)")

    # Filter subjects by class balance
    # onoff is binary (0.0=OFFTASK, 100.0=ONTASK); do NOT use a global median
    # as a split boundary — with binary data median equals the majority class
    # value, making (x > median).mean() = 0 for everyone when one class
    # dominates globally, which would exclude all subjects.
    if min_minority_ratio is not None and "onoff" in pivot.columns:
        subject_balance = pivot.groupby("subject")["onoff"].apply(
            lambda x: min(
                (x == 0.0).mean(),    # OFFTASK ratio
                (x == 100.0).mean(),  # ONTASK ratio
            )
        )
        valid_subjects = subject_balance[
            subject_balance >= min_minority_ratio
        ].index
        n_excluded = pivot["subject"].nunique() - len(valid_subjects)
        pivot = pivot[pivot["subject"].isin(valid_subjects)]
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} subjects (class imbalance)")

    print(
        f"  Ready for LMM: {len(pivot)} probes, "
        f"{pivot['subject'].nunique()} subjects, "
        f"{len(connection_ids)} connections"
    )

    return pivot, connection_ids
