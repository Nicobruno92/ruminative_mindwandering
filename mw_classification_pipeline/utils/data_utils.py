"""
Mind-Wandering Classification Pipeline — Data Utilities.

Handles loading EEG features from Junifer aggregated probe CSV files
and preparing them for classification analysis.

Data format (from aggregate_markers_by_probe.py):
    CSV files in long format: sub-XX_task-YY_desc-probe-NNN_TYPE_aggMarkers.csv
    Columns: subject, task, probe_number, marker_type, marker, channel, value, onoff, ...

Project: depressed_mindwandering
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    Dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    Get project root directory.

    Assumes this file is at:
        depressed_mindwandering/mw_classification_pipeline/wandering_mind_data_utils.py

    Returns
    -------
    Path
        Path to project root (depressed_mindwandering/).
    """
    return Path(__file__).resolve().parent.parent


def get_model_results_folder(config: Dict, model_name: str, model_type: str) -> str:
    """
    Generate the results folder name for a model.

    Parameters
    ----------
    config : Dict
        Configuration dictionary (may contain results_folder_pattern).
    model_name : str
        Name of the model/contrast (e.g., "ON_vs_OFF").
    model_type : str
        Type of classifier (e.g., "rf", "lr").

    Returns
    -------
    str
        Folder name for results.
    """
    pattern = config.get("results_folder_pattern", "{model_name}_{model_type}_Distribution")
    return pattern.format(model_name=model_name, model_type=model_type.upper())


# =============================================================================
# DATA LOADING
# =============================================================================

def find_probe_csvs(
    features_root: str,
    subject: str,
    task: str,
    marker_types: Union[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Find aggregated probe CSV files for a subject/task.

    File pattern: sub-XX_task-YY_desc-probe-NNN_TYPE_aggMarkers.csv
    Located in: features_root/sub-XX/eeg/junifer/

    Parameters
    ----------
    features_root : str
        Root directory for features.
    subject : str
        Subject ID (zero-padded, e.g., "04").
    task : str
        Task name (e.g., "Sart1").
    marker_types : str or List[str]
        Marker type(s) to load: "evoked", "state", "sleep".

    Returns
    -------
    Dict[str, List[str]]
        Mapping of marker_type -> list of CSV file paths.
    """
    if isinstance(marker_types, str):
        marker_types = [marker_types]

    base_dir = Path(features_root) / f"sub-{subject}" / "eeg" / "junifer"
    found = {}

    for mtype in marker_types:
        pattern = f"sub-{subject}_task-{task}_desc-probe-*_{mtype}_aggMarkers.csv"
        files = sorted(glob.glob(str(base_dir / pattern)))
        if files:
            found[mtype] = files

    return found


def load_probe_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single aggregated probe CSV file.

    Expected columns: subject, task, probe_number, marker_type, marker,
                      channel, value, onoff, + other behavioral columns.

    Parameters
    ----------
    filepath : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Probe data in long format.
    """
    df = pd.read_csv(filepath)

    # Standardize subject to zero-padded string
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str).str.zfill(2)

    return df


def load_subject_data(
    features_root: str,
    subject: str,
    task: str,
    marker_types: Union[str, List[str]],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load all probe CSVs for a subject/task combination.

    Parameters
    ----------
    features_root : str
        Root directory for features.
    subject : str
        Subject ID.
    task : str
        Task name.
    marker_types : str or List[str]
        Marker type(s) to load.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined long-format data for this subject/task.
    """
    csv_files = find_probe_csvs(features_root, subject, task, marker_types)

    if not csv_files:
        return pd.DataFrame()

    all_dfs = []
    for mtype, files in csv_files.items():
        for f in files:
            df = load_probe_csv(f)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"  Loaded sub-{subject} {task}: {len(combined)} rows, "
              f"{combined['probe_number'].nunique()} probes")

    return combined


def load_all_subjects(
    features_root: str,
    subjects: List[str],
    tasks: List[str],
    marker_types: Union[str, List[str]],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate data across all subjects and tasks.

    Parameters
    ----------
    features_root : str
        Root directory for features.
    subjects : List[str]
        List of subject IDs (zero-padded).
    tasks : List[str]
        List of task names.
    marker_types : str or List[str]
        Marker type(s) to load.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined long-format data across all subjects.
    """
    all_dfs = []

    for subject in subjects:
        for task in tasks:
            df = load_subject_data(features_root, subject, task, marker_types, verbose)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        if verbose:
            print("WARNING: No data loaded! Check paths and file existence.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Standardize types
    combined["subject"] = combined["subject"].astype(str).str.zfill(2)
    combined["task"] = combined["task"].astype(str)
    if "probe_number" in combined.columns:
        combined["probe_number"] = combined["probe_number"].astype(int)

    if verbose:
        n_subjects = combined["subject"].nunique()
        n_probes = combined.groupby(["subject", "task"])["probe_number"].nunique().sum()
        print(f"\nTotal: {len(combined)} rows from {n_subjects} subjects, "
              f"{n_probes} total probes")

    return combined


# =============================================================================
# PIVOT LONG → WIDE FORMAT
# =============================================================================

def pivot_to_wide(
    df_long: pd.DataFrame,
    data_format: str = "per_channel",
) -> pd.DataFrame:
    """
    Pivot long-format probe data to wide format for classification.

    Long format: one row per (probe, marker, channel) combination
    Wide format (per_channel): one row per probe, columns = {channel}_{marker}_value
    Wide format (aggregated): one row per probe, columns = {marker}_value
                              (requires pre-aggregated ROI data)

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format data with columns: subject, task, probe_number,
        marker_type, marker, channel, value, onoff, ...
    data_format : str
        "per_channel" for channel-level features,
        "aggregated" for ROI-aggregated features.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per probe.
    """
    # Identify probe-level metadata columns (same for all rows of a probe)
    probe_keys = ["subject", "task", "probe_number"]
    behavioral_cols = ["onoff", "valence", "selfother", "confidence", "time"]
    behavioral_cols = [c for c in behavioral_cols if c in df_long.columns]
    meta_cols = probe_keys + behavioral_cols

    # Create feature name column
    if data_format == "per_channel" and "channel" in df_long.columns:
        df_long = df_long.copy()
        df_long["feature_name"] = df_long["channel"] + "_" + df_long["marker"]
    else:
        df_long = df_long.copy()
        df_long["feature_name"] = df_long["marker"]

    # Extract metadata (one row per probe)
    df_meta = df_long.groupby(probe_keys)[behavioral_cols].first().reset_index()

    # Pivot: rows = probes, columns = features, values = marker value
    df_wide = df_long.pivot_table(
        index=probe_keys,
        columns="feature_name",
        values="value",
        aggfunc="first",  # Should be unique per (probe, feature)
    ).reset_index()

    # Flatten multi-level columns
    df_wide.columns = [c if isinstance(c, str) else str(c) for c in df_wide.columns]

    # Merge metadata back
    df_wide = pd.merge(df_wide, df_meta, on=probe_keys, how="left")

    return df_wide


# =============================================================================
# LABEL CONTRAST CREATION
# =============================================================================

def create_label_contrast(
    df: pd.DataFrame,
    contrast_config: Dict,
) -> pd.DataFrame:
    """
    Create binary target column from a continuous label score (e.g., onoff).

    Supports several splitting methods driven by 'split_method' or legacy keys:
    - 'global_median': Split by median across all subjects.
    - 'within_subject_median': Split by median calculated per subject.
    - 'threshold': Fixed threshold split (e.g. > 50).
    - 'extreme_groups': Takes top and bottom groups, drops middle.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with the target label column.
    contrast_config : Dict
        Contrast configuration from config.yaml.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'target' column. Rows outside contrast are dropped.
    """
    # Allow column_name as fallback to label_source for compatibility
    label_col = contrast_config.get("label_source", contrast_config.get("column_name", "onoff"))

    if label_col not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Label column '{label_col}' not found. Available: {available}"
        )

    df = df.copy()
    
    split_method = contrast_config.get("split_method", None)
    positive_above = contrast_config.get("positive_above", True)

    # Mode 1: Global Median
    if split_method == "global_median" or split_method == "median":
        median_val = df[label_col].median()
        if positive_above:
            df["target"] = (df[label_col] > median_val).astype(int)
        else:
            df["target"] = (df[label_col] <= median_val).astype(int)
        print(f"  Global median split: {label_col} {'>' if positive_above else '<='} {median_val:.2f}")

    # Mode 2: Within-Subject Median
    elif split_method == "within_subject_median":
        if "subject" not in df.columns:
            raise ValueError(
                "Cannot compute within-subject median because 'subject' column is missing."
            )
        subject_medians = df.groupby("subject")[label_col].transform("median")
        if positive_above:
            df["target"] = (df[label_col] > subject_medians).astype(int)
        else:
            df["target"] = (df[label_col] <= subject_medians).astype(int)
        print(f"  Within-subject median split: {label_col} {'>' if positive_above else '<='} subject_median")

    # Mode 3: Simple fixed threshold
    elif split_method == "threshold" or ("threshold" in contrast_config and "threshold_low" not in contrast_config):
        threshold = contrast_config.get("threshold", 50)
        if positive_above:
            df["target"] = (df[label_col] > threshold).astype(int)
        else:
            df["target"] = (df[label_col] <= threshold).astype(int)
        print(f"  Threshold binarization: {label_col} {'>' if positive_above else '<='} {threshold}")

    # Mode 4: Extreme groups (exclude middle range)
    elif split_method == "extreme_groups" or ("threshold_low" in contrast_config and "threshold_high" in contrast_config):
        low = contrast_config["threshold_low"]
        high = contrast_config["threshold_high"]

        df["target"] = np.nan
        df.loc[df[label_col] <= low, "target"] = 0
        df.loc[df[label_col] >= high, "target"] = 1

        n_before = len(df)
        df = df.dropna(subset=["target"])
        n_excluded = n_before - len(df)

        print(f"  Extreme groups: {label_col} <= {low} → OFF, "
              f"{label_col} >= {high} → ON")
        print(f"  Excluded {n_excluded} probes in middle range ({low}-{high})")

        df["target"] = df["target"].astype(int)

    else:
        raise ValueError(
            f"Invalid contrast config: {contrast_config}. "
            f"Specify a valid 'split_method' (global_median, within_subject_median, "
            f"threshold, extreme_groups) or legacy threshold parameters."
        )

    return df


def get_class_distribution(df: pd.DataFrame, target_col: str = "target") -> Dict:
    """
    Get class distribution statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target column.
    target_col : str
        Name of target column.

    Returns
    -------
    Dict
        Distribution statistics.
    """
    counts = df[target_col].value_counts()
    total = len(df)

    return {
        "n_positive": counts.get(1, 0),
        "n_negative": counts.get(0, 0),
        "total": total,
        "positive_fraction": counts.get(1, 0) / total if total > 0 else 0,
        "negative_fraction": counts.get(0, 0) / total if total > 0 else 0,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def get_feature_columns(
    df: pd.DataFrame,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify feature columns (exclude metadata and label columns).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format feature DataFrame.
    exclude_patterns : List[str], optional
        Additional column name patterns to exclude.

    Returns
    -------
    List[str]
        List of feature column names.
    """
    # Metadata columns to exclude
    metadata_cols = {
        "subject", "task", "probe_number", "channel", "marker", "marker_type",
        "value", "onoff", "valence", "selfother", "confidence", "time",
        "label", "ontask_label", "content", "target", "sample_id",
        "subject_id", "observation_id", "n_epochs", "feature_name",
    }

    # Add custom patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            metadata_cols.update(
                c for c in df.columns if pattern in c.lower()
            )

    # Numeric columns not in metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]

    return feature_cols


# =============================================================================
# PARTICIPANT FILTERING
# =============================================================================

def filter_participants_by_balance(
    df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    min_minority_fraction: float,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter out participants with extreme class imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame.
    subject_col : str
        Column name for subject identifier.
    target_col : str
        Column name for target variable.
    min_minority_fraction : float
        Minimum fraction of minority class required (0.0-0.5).
    verbose : bool
        Print filtering information.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if min_minority_fraction <= 0:
        return df

    subjects_to_drop = []

    for subject, group in df.groupby(subject_col):
        counts = group[target_col].value_counts(normalize=True)

        if len(counts) < 2:
            subjects_to_drop.append((subject, 0.0, "Only one class"))
        else:
            minority_fraction = counts.min()
            if minority_fraction < min_minority_fraction:
                subjects_to_drop.append(
                    (subject, minority_fraction, f"Minority {minority_fraction:.2%}")
                )

    if not subjects_to_drop:
        return df

    if verbose:
        print(f"\nFiltering participants with minority class < "
              f"{min_minority_fraction:.0%}:")
        for subject, frac, reason in subjects_to_drop:
            print(f"  Dropping sub-{subject}: {reason}")

    drop_ids = [s[0] for s in subjects_to_drop]
    filtered_df = df[~df[subject_col].isin(drop_ids)].copy()

    if verbose:
        n_kept = df[subject_col].nunique() - len(drop_ids)
        print(f"  Kept {len(filtered_df)} samples from {n_kept} subjects")

    return filtered_df


def filter_participants_by_sample_count(
    df: pd.DataFrame,
    subject_col: str,
    min_samples: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter out participants with fewer than minimum number of samples.

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame.
    subject_col : str
        Column name for subject identifier.
    min_samples : int
        Minimum number of samples required per subject.
    verbose : bool
        Print filtering information.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if min_samples <= 0:
        return df

    subjects_to_drop = []

    for subject, group in df.groupby(subject_col):
        n_samples = len(group)
        if n_samples < min_samples:
            subjects_to_drop.append((subject, n_samples))

    if not subjects_to_drop:
        return df

    if verbose:
        print(f"\nFiltering participants with < {min_samples} samples:")
        for subject, n_samples in subjects_to_drop:
            print(f"  Dropping sub-{subject}: {n_samples} samples")

    drop_ids = [s[0] for s in subjects_to_drop]
    filtered_df = df[~df[subject_col].isin(drop_ids)].copy()

    if verbose:
        n_kept = df[subject_col].nunique() - len(drop_ids)
        print(f"  Kept {len(filtered_df)} samples from {n_kept} subjects")

    return filtered_df


# =============================================================================
# MAIN DATA PREPARATION
# =============================================================================

def prepare_data_for_contrast(
    config: Dict,
    contrast_name: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Main entry point: load data and prepare for classification.

    Parameters
    ----------
    config : Dict
        Configuration dictionary.
    contrast_name : str
        Name of contrast from label_contrasts config.
    verbose : bool
        Print progress information.

    Returns
    -------
    Tuple containing:
        - df_prepared: Full prepared DataFrame
        - X: Feature matrix (pd.DataFrame)
        - y: Target vector (pd.Series)
        - groups: Subject grouping for CV (pd.Series)
        - feature_cols: List of feature column names
    """
    # Extract config values
    features_root = config["data_paths"]["features_root"]
    subjects = config.get("subjects")
    tasks = config["tasks"]
    epoch_types = config.get("epoch_types", ["state"])
    data_format = config.get("data_format", "per_channel")
    min_minority_ratio = config.get("min_minority_ratio", 0.15)
    min_samples = config.get("min_samples", 0)

    # Auto-detect subjects if not specified
    if subjects is None:
        project_root = get_project_root()
        abs_features_root = features_root
        if not os.path.isabs(features_root):
            abs_features_root = str(project_root / features_root)

        if verbose:
            print("No subjects specified, scanning features directory...")
        subject_dirs = sorted(glob.glob(os.path.join(abs_features_root, "sub-*")))
        subjects = [os.path.basename(d).replace("sub-", "") for d in subject_dirs]
        if verbose:
            print(f"  Found {len(subjects)} subjects: {subjects}")

    # Validate contrast
    if contrast_name not in config["label_contrasts"]:
        available = list(config["label_contrasts"].keys())
        raise ValueError(f"Contrast '{contrast_name}' not found. Available: {available}")

    contrast_config = config["label_contrasts"][contrast_name]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Preparing data for contrast: {contrast_name}")
        print(f"  Marker types: {epoch_types}")
        print(f"  Data format: {data_format}")
        print(f"  Contrast config: {contrast_config}")
        print(f"{'='*60}\n")

    # Resolve features_root
    project_root = get_project_root()
    if not os.path.isabs(features_root):
        features_root = str(project_root / features_root)

    # Load all data in long format
    if verbose:
        print("Loading aggregated probe marker CSVs...")

    df_long = load_all_subjects(features_root, subjects, tasks, epoch_types, verbose)

    if df_long.empty:
        raise ValueError("No data loaded! Check paths and file existence.")

    # Pivot to wide format (one row per probe)
    if verbose:
        print(f"\nPivoting to wide format ({data_format})...")

    df_wide = pivot_to_wide(df_long, data_format)

    if verbose:
        print(f"  Wide format: {df_wide.shape[0]} probes × {df_wide.shape[1]} columns")

    # Create binary target
    if verbose:
        print(f"\nCreating binary target for {contrast_name}...")

    df_wide = create_label_contrast(df_wide, contrast_config)

    # Check class distribution
    dist = get_class_distribution(df_wide)
    if verbose:
        print(f"  Class 0 (OFF): {dist['n_negative']} ({dist['negative_fraction']:.1%})")
        print(f"  Class 1 (ON):  {dist['n_positive']} ({dist['positive_fraction']:.1%})")

    # Filter participants by minimum sample count
    if min_samples > 0:
        df_wide = filter_participants_by_sample_count(
            df_wide, "subject", min_samples, verbose
        )

    # Filter participants by class balance
    df_wide = filter_participants_by_balance(
        df_wide, "subject", "target", min_minority_ratio, verbose
    )

    if df_wide.empty:
        raise ValueError("No data remaining after participant filtering!")

    # Get feature columns
    feature_cols = get_feature_columns(df_wide)

    if verbose:
        print(f"\nIdentified {len(feature_cols)} feature columns")
        if len(feature_cols) > 10:
            print(f"  Examples: {feature_cols[:5]} ... {feature_cols[-5:]}")
        else:
            print(f"  Features: {feature_cols}")

    # Create sample_id for tracking
    df_wide["sample_id"] = (
        df_wide["subject"].astype(str) + "_"
        + df_wide["task"].astype(str) + "_"
        + df_wide["probe_number"].astype(str)
    )

    # Extract X, y, groups
    X = df_wide[feature_cols].copy()
    y = df_wide["target"].copy()
    groups = df_wide["subject"].copy()

    # Drop features that are all NaN
    constant_nan_cols = X.columns[X.isna().all()]
    if len(constant_nan_cols) > 0:
        if verbose:
            print(f"\n[INFO] Dropping {len(constant_nan_cols)} all-NaN features")
        X = X.drop(columns=constant_nan_cols)
        feature_cols = [c for c in feature_cols if c not in constant_nan_cols]

    # Drop samples with any remaining NaN
    n_before = len(X)
    invalid_mask = X.isna().any(axis=1)

    if invalid_mask.any() and verbose:
        dropped_df = df_wide[invalid_mask][["subject", "task", "probe_number"]]
        print(f"\n[WARNING] Dropping {invalid_mask.sum()} samples with missing features:")
        for _, row in dropped_df.head(10).iterrows():
            print(f"  - sub-{row['subject']} {row['task']} probe-{row['probe_number']}")

    valid_mask = ~invalid_mask
    X = X[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]
    df_prepared = df_wide[valid_mask].copy()
    n_after = len(X)

    if n_before > n_after and verbose:
        print(f"  Dropped {n_before - n_after} samples with missing features")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final dataset:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Subjects: {groups.nunique()}")
        print(f"  Class distribution: {y.value_counts().to_dict()}")
        print(f"{'='*60}\n")

    return df_prepared, X, y, groups, feature_cols


def get_features_target_groups(
    df_prepared: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Extract X, y, groups from prepared data.

    Compatibility function matching pipeline signature.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared DataFrame from prepare_data_for_contrast.
    feature_cols : List[str]
        List of feature column names.

    Returns
    -------
    Tuple containing:
        - X: Feature matrix
        - y: Target vector
        - groups: Subject groups for CV
        - class_info: Dictionary with class metadata
    """
    X = df_prepared[feature_cols].copy()
    y = df_prepared["target"].copy()
    groups = df_prepared["subject"].copy()

    class_info = {
        "classes_flipped": False,
        "positive_class_name": "ON-task (1)",
        "negative_class_name": "OFF-task (0)",
    }

    return X, y, groups, class_info


# =============================================================================
# CONVENIENCE
# =============================================================================

def load_and_prepare_from_config_path(
    config_path: str,
    contrast_name: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict]:
    """
    Load config and prepare data in one call.

    Parameters
    ----------
    config_path : str
        Path to config YAML file.
    contrast_name : str, optional
        Contrast to prepare. If None, uses config["contrast"].
    verbose : bool
        Print progress.

    Returns
    -------
    Tuple containing:
        - df_prepared, X, y, groups, feature_cols, config
    """
    config = load_config(config_path)

    if contrast_name is None:
        contrast_name = config.get("contrast", "ON_vs_OFF")

    df_prepared, X, y, groups, feature_cols = prepare_data_for_contrast(
        config, contrast_name, verbose=verbose
    )

    return df_prepared, X, y, groups, feature_cols, config
