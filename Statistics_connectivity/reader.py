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

# Separator used in channel pair column names
PAIR_SEPARATOR = "--"


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
    pattern = "**/junifer_aggregated/*_connectivity.csv"
    csv_files = sorted(features_path.glob(pattern))

    if verbose:
        print(f"Found {len(csv_files)} connectivity CSV files")

    assert len(csv_files) > 0, (
        f"No connectivity CSV files found in {features_root}"
    )

    # Filter files
    filtered_files = _filter_files(csv_files, subjects, tasks, epoch_types)

    if verbose:
        print(f"After filtering: {len(filtered_files)} files to load")

    # Load and combine
    all_rows = []
    for file_path in filtered_files:
        rows = _parse_connectivity_csv(file_path, bands)
        all_rows.extend(rows)

    if verbose:
        print(f"  Loaded {len(all_rows)} connection records")

    combined_df = pd.DataFrame(all_rows)

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
) -> List[Dict]:
    """
    Parse a single connectivity CSV file into long-format records.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.
    bands : Optional[List[str]]
        Bands to include. None = all.

    Returns
    -------
    List[Dict]
        List of records, each with: subject, task, probe_number, label,
        n_epochs, onoff, epoch_type, band, connection_id, wsmi_value
    """
    df = pd.read_csv(file_path)

    # Extract metadata from columns
    subject = str(df["subject"].iloc[0])
    task = str(df["task"].iloc[0])
    probe_number = int(df["probe_number"].iloc[0])
    label = str(df["label"].iloc[0])
    n_epochs = int(df["n_epochs"].iloc[0])

    # Extract onoff from metadata columns (if present)
    onoff_val = float(df["onoff"].iloc[0]) if "onoff" in df.columns else np.nan

    # Extract additional behavioral variables
    valence = float(df["valence"].iloc[0]) if "valence" in df.columns else np.nan
    selfother = float(df["selfother"].iloc[0]) if "selfother" in df.columns else np.nan
    confidence = float(df["confidence"].iloc[0]) if "confidence" in df.columns else np.nan
    time_val = float(df["time"].iloc[0]) if "time" in df.columns else np.nan

    # Extract epoch type from filename
    fname = file_path.name.replace(".csv", "")
    parts = fname.split("_")
    epoch_type = "unknown"
    for i, p in enumerate(parts):
        if p == "connectivity" and i > 0:
            epoch_type = parts[i - 1] + "_connectivity"
            break

    # Find wSMI marker columns: wsmi_{band}_pairs_trimmean (or similar)
    # The "channel" column in these files contains the pair identifier (e.g., "Fp1--F3")
    marker_cols = [
        c for c in df.columns
        if c.startswith("wsmi_") and "pairs" in c
    ]

    rows = []
    for _, row in df.iterrows():
        connection_id = str(row["channel"])

        for col in marker_cols:
            # Extract band from column name: wsmi_theta_pairs_trimmean → theta
            band = _extract_band_from_column(col)
            if bands is not None and band not in bands:
                continue

            wsmi_value = float(row[col])

            rows.append({
                "subject": subject,
                "task": task,
                "probe_number": probe_number,
                "label": label,
                "n_epochs": n_epochs,
                "onoff": onoff_val,
                "valence": valence,
                "selfother": selfother,
                "confidence": confidence,
                "time": time_val,
                "epoch_type": epoch_type,
                "band": band,
                "connection_id": connection_id,
                "wsmi_value": wsmi_value,
            })

    return rows


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

    result = (
        df.groupby(group_cols, as_index=False)
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
    metadata_cols = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "valence", "selfother", "confidence", "time",
    ]
    metadata_cols = [c for c in metadata_cols if c in filtered.columns]

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
    if min_minority_ratio is not None and "onoff" in pivot.columns:
        median_onoff = pivot["onoff"].median()
        subject_balance = pivot.groupby("subject")["onoff"].apply(
            lambda x: min(
                (x <= median_onoff).mean(),
                (x > median_onoff).mean(),
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
