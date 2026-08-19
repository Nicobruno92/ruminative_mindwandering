"""
Reader module for connectivity statistics pipeline.

Loads wSMI connectivity pair data from aggregated CSV files and optionally
aggregates channel-pair values into ROI-pair values.

Upstream contract
-----------------
Pre-probe distance filtering (CLAUDE.md: distance −5 to −1) is applied
*upstream* in ``junifer_markers/2.aggregate_probes/aggregate_markers_by_probe.py``
via the ``evoked_distance_min/max`` config (default 0..4 = the 5 epochs
closest to the probe). The connectivity CSVs read here therefore already
encode the correct probe window; this module trusts that contract and does
not re-filter by distance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations_with_replacement


# =============================================================================
# CONSTANTS
# =============================================================================

# Separator used in channel pair column names and ROI pair names.
# The Junifer pipeline writes pairs as "Ch1-Ch2" (single hyphen).
# EEG channel names (Fp1, AF3, FC1, …) never contain hyphens, and
# ROI names use underscores, so a single hyphen is unambiguous.
PAIR_SEPARATOR = "-"

# SART → integer index mapping used to build ``time_on_task``.
# Kept here because the aggregated probe CSVs do not carry this mapping
# directly (the task name is what's stored). If the experiment ever changes
# the SART labelling this mapping must be updated.
SART_TASK_INDEX: Dict[str, int] = {
    "Sart1": 1,
    "Sart2": 2,
    "Sart3": 3,
    "Sart4": 4,
}

# Number of probes per SART block in the CYBERSART experimental design.
# Used to convert (task, probe_number) → cumulative time_on_task index.
PROBES_PER_SART: int = 15


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
    
    # Calculate cumulative time-on-task (probe index across the four Sart blocks).
    # Uses module-level constants so the mapping is in one place and not magic.
    sart_number = SART_TASK_INDEX.get(task)
    if sart_number is not None:
        time_on_task_val = int(probe_number + PROBES_PER_SART * (sart_number - 1))
    else:
        time_on_task_val = np.nan

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
    result["time_on_task"] = time_on_task_val
    result["epoch_type"] = epoch_type

    out_cols = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "valence", "selfother", "confidence", "time", "time_on_task",
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

    # Group by all metadata + ROI pair and average wSMI.
    # Includes optional PCA components so they propagate through ROI aggregation
    # when merge_pca_results was called upstream.
    group_cols = [
        "subject", "task", "probe_number", "label", "n_epochs",
        "onoff", "valence", "selfother", "confidence", "time", "time_on_task",
        "PC1", "PC2", "PC3",
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
# QA FILTERING / PCA MERGE / NORMALIZATION
# =============================================================================

def apply_qa_filter(
    df: pd.DataFrame,
    qa_summary_path: str,
    epoch_types: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop (subject, task, epoch_type) tuples that failed preprocessing QA.

    The QA summary CSV ``qa_summary_path`` has one row per
    (subject, task, epoch_type) with a boolean ``passed`` column written by
    the preprocessing pipeline. The connectivity ``epoch_type`` column is
    suffixed with ``_connectivity`` (e.g. ``sleep_connectivity``); the QA
    table uses the bare root (e.g. ``sleep``) so we strip the suffix before
    matching.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data with columns ``subject``, ``task``,
        ``epoch_type``.
    qa_summary_path : str
        Path to ``qa_summary.csv`` (boolean ``passed`` column required).
    epoch_types : Optional[List[str]]
        Epoch types we care about (e.g. ``["sleep_connectivity"]``). Used
        only for logging; the join is on the actual ``epoch_type`` column.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    pd.DataFrame
        Filtered long-format data (rows where the corresponding QA row is
        ``passed = True``).
    """
    qa = pd.read_csv(qa_summary_path)
    assert {"subject", "task", "epoch_type", "passed"}.issubset(qa.columns), (
        f"qa_summary missing required columns; got {list(qa.columns)}"
    )

    # Harmonise key types: connectivity df uses zero-padded subject strings,
    # QA file stores subject as int. Cast both sides to zero-padded str.
    qa = qa.copy()
    qa["subject"] = qa["subject"].astype(int).map(lambda s: f"{s:02d}")
    qa["task"] = qa["task"].astype(str)
    qa["epoch_type_root"] = qa["epoch_type"].astype(str)

    # Map connectivity-side epoch_type ("sleep_connectivity") to root ("sleep")
    df = df.copy()
    df["epoch_type_root"] = (
        df["epoch_type"].astype(str).str.replace("_connectivity", "", regex=False)
    )

    failed = qa.loc[~qa["passed"].astype(bool), ["subject", "task", "epoch_type_root"]]
    if failed.empty:
        if verbose:
            print("  QA filter: 0 (subject, task, epoch_type) tuples flagged failed")
        return df.drop(columns=["epoch_type_root"])

    # Anti-join: drop rows whose key matches any failed tuple
    keys_failed = set(map(tuple, failed.values))
    n_before = len(df)
    keep_mask = ~df.apply(
        lambda r: (r["subject"], r["task"], r["epoch_type_root"]) in keys_failed,
        axis=1,
    )
    df_out = df.loc[keep_mask].drop(columns=["epoch_type_root"]).reset_index(drop=True)
    n_after = len(df_out)

    if verbose:
        print(
            f"  QA filter: dropped {n_before - n_after}/{n_before} rows "
            f"({len(keys_failed)} failed (subject,task,epoch) tuples)"
        )

    return df_out


def merge_pca_results(
    df: pd.DataFrame,
    pca_results_path: str,
    pca_columns: Tuple[str, ...] = ("PC1", "PC2", "PC3"),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Left-merge probe-level PCA components into the connectivity long df.

    PCA is computed upstream on the behavioural probe-content responses; this
    function attaches PC scores to each (subject, task, probe_number) row so
    they are available as LMM predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data.
    pca_results_path : str
        Path to ``pca_results.csv`` (must contain ``subject_id``/``subject``,
        ``task``, ``probe_number`` and the PC columns).
    pca_columns : Tuple[str, ...]
        Names of the PC columns to bring in.
    verbose : bool
        Print merge statistics.

    Returns
    -------
    pd.DataFrame
        Same as input with ``pca_columns`` added; rows with no matching PCA
        entry get NaN in the new columns (and will be filtered downstream by
        ``prepare_connectivity_for_lmm``).
    """
    pca = pd.read_csv(pca_results_path)
    subj_col = "subject_id" if "subject_id" in pca.columns else "subject"
    needed = [subj_col, "task", "probe_number", *pca_columns]
    missing = [c for c in needed if c not in pca.columns]
    assert not missing, f"PCA file missing columns {missing}; got {list(pca.columns)}"

    pca = pca[needed].rename(columns={subj_col: "subject"}).copy()
    # Harmonise keys: zero-padded string subject + integer probe number
    pca["subject"] = pca["subject"].astype(int).map(lambda s: f"{s:02d}")
    pca["task"] = pca["task"].astype(str)
    pca["probe_number"] = pca["probe_number"].astype(float).astype(int)
    # Some PCA rows can be duplicated; collapse to one row per probe
    pca = pca.drop_duplicates(subset=["subject", "task", "probe_number"], keep="first")

    df = df.copy()
    df["probe_number"] = df["probe_number"].astype(int)
    merged = df.merge(pca, on=["subject", "task", "probe_number"], how="left")

    if verbose:
        n_total = len(merged)
        n_matched = int(merged[list(pca_columns)[0]].notna().sum())
        print(
            f"  PCA merge: {n_matched}/{n_total} rows matched "
            f"({100*n_matched/n_total:.1f}%) for {list(pca_columns)}"
        )

    return merged


def normalize_wsmi_by_subject(
    df: pd.DataFrame,
    method: str = "zscore",
    channel_wise: bool = False,
    subject_column: str = "subject",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Normalize the ``wsmi_value`` column within each subject.

    Z-scoring per subject removes between-subject mean/scale differences in
    raw wSMI before LMM, complementing (not replacing) the random intercept.
    With ``channel_wise=True`` the z-score is computed per
    (subject × connection_id), i.e. each connection has its own per-subject
    distribution; this is the more aggressive option and usually only
    sensible if you expect distinct per-connection scales.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data.
    method : str
        ``"zscore"`` (currently the only supported method).
    channel_wise : bool
        If True, normalize per (subject, connection_id) instead of per subject.
    subject_column : str
        Name of the subject column.
    verbose : bool
        Print a short summary.

    Returns
    -------
    pd.DataFrame
        Same shape as input with ``wsmi_value`` z-scored.
    """
    assert method == "zscore", f"Only 'zscore' supported, got {method!r}"
    df = df.copy()
    group_cols = [subject_column]
    if channel_wise:
        group_cols.append("connection_id")

    grouped = df.groupby(group_cols)["wsmi_value"]
    mu = grouped.transform("mean")
    sd = grouped.transform("std")
    # Avoid divide-by-zero: groups with 0 variance get the centred value (0).
    sd = sd.where(sd > 0, np.nan)
    z = (df["wsmi_value"] - mu) / sd
    df["wsmi_value"] = z.fillna(0.0)

    if verbose:
        scope = "subject × connection" if channel_wise else "subject"
        print(f"  wSMI normalized (zscore) per {scope}")
    return df


def add_quadratic_features(
    df: pd.DataFrame,
    quadratic_features: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add orthogonalized quadratic features as new columns in the DataFrame.

    For each ``base_col → derived_col`` pair:

    1. Compute ``raw_sq = (base_col - 50)^2 / 50`` on unique probe rows.
    2. Fit OLS: ``raw_sq ~ base_col`` to capture the marginal linear component.
    3. ``derived_col = raw_sq - (intercept + slope * base_col)`` — residual is
       orthogonal to the linear predictor at the population level.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level DataFrame (one row per probe, not long-format).
        Must contain the base predictor columns (e.g. ``valence``, ``time``).
    quadratic_features : dict
        Mapping ``{base_col: derived_col}``,
        e.g. ``{"valence": "valence_sq", "time": "time_sq"}``.
    verbose : bool
        Whether to print OLS fit statistics and residual correlations.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the derived quadratic columns appended.
    """
    from scipy import stats as scipy_stats

    df_out = df.copy()

    # For connectivity data, rows may already be unique probes per connection;
    # use subject/task/probe_number to deduplicate for OLS fitting when present.
    key_cols = [c for c in ('subject', 'task', 'probe_number') if c in df_out.columns]
    df_unique = df_out.drop_duplicates(subset=key_cols) if key_cols else df_out

    ols_fits: dict = {}

    for base_col, sq_col in quadratic_features.items():
        if base_col not in df_out.columns:
            raise ValueError(
                f"Base column '{base_col}' not found in DataFrame. "
                f"Available columns: {list(df_out.columns)}"
            )

        base_vals = df_unique[base_col].values.astype(float)
        valid_mask = ~np.isnan(base_vals)
        raw_sq = (base_vals - 50.0) ** 2 / 50.0

        slope, intercept, r_val, _, _ = scipy_stats.linregress(
            base_vals[valid_mask], raw_sq[valid_mask]
        )
        ols_fits[base_col] = (slope, intercept)

        if verbose:
            print(
                f"  {sq_col}: OLS slope={slope:.4f}, intercept={intercept:.4f}, "
                f"r={r_val:.4f} with {base_col} (n={int(valid_mask.sum())} unique probes)"
            )

    for base_col, sq_col in quadratic_features.items():
        if base_col not in df_out.columns:
            continue
        slope, intercept = ols_fits[base_col]
        base_all = df_out[base_col].values.astype(float)
        raw_sq_all = (base_all - 50.0) ** 2 / 50.0
        residual = raw_sq_all - (intercept + slope * base_all)
        residual[np.isnan(base_all)] = np.nan
        df_out[sq_col] = residual

        if verbose:
            valid = ~np.isnan(base_all)
            r_check = float(np.corrcoef(base_all[valid], residual[valid])[0, 1])
            print(f"  {sq_col}: residual r with {base_col} = {r_check:.6f} (expected ~0)")

    return df_out


def normalize_predictors(
    df: pd.DataFrame,
    predictors: List[str],
    method: str = "zscore",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Z-score continuous predictors across the full sample.

    Putting predictors on a common scale (a) makes fixed-effect coefficients
    comparable across covariates and (b) improves LMM optimiser conditioning.
    Categorical-binary predictors (e.g. ``onoff`` coded 0/100) are left alone:
    z-scoring would only change the coefficient interpretation, not the test.

    Parameters
    ----------
    df : pd.DataFrame
        Long or wide DataFrame containing the predictor columns.
    predictors : List[str]
        Predictor column names to z-score.
    method : str
        ``"zscore"`` (currently the only supported method).
    verbose : bool
        Print which columns were z-scored.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with the listed predictor columns z-scored in place
        (NaN-safe: NaNs are preserved, ignored in the mean/std).
    """
    assert method == "zscore", f"Only 'zscore' supported, got {method!r}"
    df = df.copy()
    z_done = []
    for col in predictors:
        if col not in df.columns:
            continue
        x = df[col].astype(float)
        if x.notna().sum() < 2:
            continue
        # Skip binary 0/100 (onoff) — z-scoring is a relabel, not informative.
        unique_vals = x.dropna().unique()
        if len(unique_vals) <= 2:
            continue
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if not (sd and np.isfinite(sd) and sd > 0):
            continue
        df[col] = (x - mu) / sd
        z_done.append(col)
    if verbose and z_done:
        print(f"  Predictors z-scored: {z_done}")
    return df


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
        "time_on_task",
        # PCA components merged in by reader.merge_pca_results — carry them
        # through the pivot so they remain available as LMM predictors.
        "PC1", "PC2", "PC3",
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
        unique_vals = pivot["onoff"].dropna().unique()
        if len(unique_vals) > 10:
            print(f"  Skipping class imbalance filter: 'onoff' is continuous ({len(unique_vals)} unique values)")
        else:
            # For discrete variables (like binary onoff), check if minority class has enough trials
            subject_balance = pivot.groupby("subject")["onoff"].apply(
                lambda x: x.value_counts(normalize=True).min() if len(x.dropna().unique()) > 1 else 0.0
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
