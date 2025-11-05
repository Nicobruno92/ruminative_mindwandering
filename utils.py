"""Utility functions for EEG markers."""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


def filter_to_eeg_channels(data_obj):
    """Filter data to only EEG channels (supports both EGI and standard naming).

    Parameters
    ----------
    data_obj : mne.Epochs or mne.Raw
        MNE data object

    Returns
    -------
    filtered_data : mne.Epochs or mne.Raw
        Data with only EEG channels
    eeg_channel_names : list
        Names of EEG channels
    eeg_channel_indices : list
        Indices of EEG channels
    """
    # Try EGI naming convention first (E1, E2, ..., E256)
    egi_channel_indices = [
        i
        for i, ch in enumerate(data_obj.ch_names)
        if ch.startswith("E") and ch[1:].isdigit()
    ]

    if egi_channel_indices:
        # Use EGI channels if found
        eeg_channel_indices = egi_channel_indices
        eeg_channel_names = [data_obj.ch_names[i] for i in eeg_channel_indices]
    else:
        # Fall back to MNE's channel type detection for standard EEG names
        import mne

        eeg_picks = mne.pick_types(data_obj.info, eeg=True, exclude=[])
        eeg_channel_indices = eeg_picks.tolist()
        eeg_channel_names = [data_obj.ch_names[i] for i in eeg_channel_indices]

    # Filter data to only EEG channels
    filtered_data = data_obj.copy().pick(eeg_channel_names)

    return filtered_data, eeg_channel_names, eeg_channel_indices


def check_indices(indices):
    """Check and format connectivity indices.

    Parameters
    ----------
    indices : tuple of array_like or None
        Indices specifying connectivity pairs

    Returns
    -------
    tuple of np.ndarray
        Validated indices as (sources, targets)
    """
    if indices is None:
        return None
    return (np.array(indices[0]), np.array(indices[1]))


def get_roi_mapping() -> Dict[str, List[str]]:
    """Get mapping of ROI names to electrode names.

    Returns standard 10-20 system ROI definitions and ICM-specific ROIs.

    Returns
    -------
    dict
        Mapping from ROI names to lists of electrode names.
    """
    roi_mapping = {
        # Standard 10-20 system ROIs
        # Frontal
        "Fp": ["Fp1", "Fp2"],
        "F": ["F3", "F4", "F7", "F8"],
        "FC": ["FC1", "FC2", "FC5", "FC6"],
        "Fz": ["Fz"],
        # Central
        "C": ["C3", "C4"],
        "CP": ["CP1", "CP2", "CP5", "CP6"],
        "Cz": ["Cz"],
        # Parietal
        "P": ["P3", "P4", "P7", "P8"],
        "Pz": ["Pz"],
        "PO": ["PO3", "PO4", "PO7", "PO8"],
        # Occipital
        "O": ["O1", "O2"],
        "Oz": ["Oz"],
        # Temporal
        "T": ["T7", "T8"],
        "TP": ["TP9", "TP10"],
        # Individual electrodes (for single electrode ROIs)
        "Fp1": ["Fp1"],
        "Fp2": ["Fp2"],
        "F3": ["F3"],
        "F4": ["F4"],
        "F7": ["F7"],
        "F8": ["F8"],
        "FC1": ["FC1"],
        "FC2": ["FC2"],
        "FC5": ["FC5"],
        "FC6": ["FC6"],
        "C3": ["C3"],
        "C4": ["C4"],
        "CP1": ["CP1"],
        "CP2": ["CP2"],
        "CP5": ["CP5"],
        "CP6": ["CP6"],
        "P3": ["P3"],
        "P4": ["P4"],
        "P7": ["P7"],
        "P8": ["P8"],
        "PO3": ["PO3"],
        "PO4": ["PO4"],
        "PO7": ["PO7"],
        "PO8": ["PO8"],
        "O1": ["O1"],
        "O2": ["O2"],
        "T7": ["T7"],
        "T8": ["T8"],
        "TP9": ["TP9"],
        "TP10": ["TP10"],
    }

    return roi_mapping


def get_icm_roi_mapping(equipment: str = "standard") -> Dict[str, List[str]]:
    """Get ICM-specific ROI mappings for different equipment configurations.

    Parameters
    ----------
    equipment : str
        Equipment configuration: 'standard', 'egi256', or 'egi128'.

    Returns
    -------
    dict
        Mapping from ICM ROI names to lists of electrode names.
    """

    if equipment == "egi256":
        # EGI 256-channel system ROI definitions for ICM Local Global paradigm
        icm_rois = {
            # Scalp ROI - all EEG channels excluding bad/reference channels
            "scalp": [
                f"E{i}"
                for i in range(1, 257)
                if i
                not in [
                    17,
                    128,
                    126,
                    127,
                    132,
                    133,
                    134,
                    135,
                    136,
                    137,
                    138,
                    139,
                    140,
                    141,
                    142,
                    143,
                    144,
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                    160,
                    161,
                    162,
                    163,
                    164,
                    165,
                    166,
                    167,
                    168,
                    169,
                    170,
                    171,
                    172,
                    173,
                    174,
                    175,
                    176,
                    177,
                    178,
                    179,
                    180,
                    181,
                    182,
                    183,
                    184,
                    185,
                    186,
                    187,
                    188,
                    189,
                    190,
                    191,
                    192,
                    193,
                    194,
                    195,
                    196,
                    197,
                    198,
                    199,
                    200,
                    201,
                    202,
                    203,
                    204,
                    205,
                    206,
                    207,
                    208,
                    209,
                    210,
                    211,
                    212,
                    213,
                    214,
                    215,
                    216,
                    217,
                    218,
                    219,
                    220,
                    221,
                    222,
                    223,
                    224,
                    225,
                    226,
                    227,
                    228,
                    229,
                    230,
                    231,
                    232,
                    233,
                    234,
                    235,
                    236,
                    237,
                    238,
                    239,
                    240,
                    241,
                    242,
                    243,
                    244,
                    245,
                    246,
                    247,
                    248,
                    249,
                    250,
                    251,
                    252,
                    253,
                    254,
                    255,
                    256,
                ]
            ],
            # CNV ROI - Frontal midline (Fz region) - matches NICE exactly
            # From NICE: np.array([6, 7, 14, 15, 16, 22, 23]) - 1 (0-indexed)
            "cnv": ["E6", "E7", "E14", "E15", "E16", "E22", "E23"],
            # CNV ROI - central and frontal-central electrodes for CNV analysis
            "old_cnv": [
                "E5",
                "E6",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E25",
                "E26",
                "E27",
                "E28",
                "E29",
                "E30",
                "E31",
                "E35",
                "E36",
                "E37",
                "E40",
                "E41",
                "E42",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
                "E117",
                "E118",
            ],
            # MMN ROI - fronto-central electrodes for mismatch negativity
            "mmn": [
                "E5",
                "E6",
                "E7",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E25",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
                "E117",
                "E118",
            ],
            # P3a ROI - fronto-central electrodes for P3a component
            "p3a": [
                "E5",
                "E6",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
            ],
            # P3b ROI - centro-parietal electrodes for P3b component
            "p3b": [
                "E7",
                "E31",
                "E37",
                "E40",
                "E41",
                "E42",
                "E47",
                "E53",
                "E54",
                "E55",
                "E60",
                "E61",
                "E62",
                "E67",
                "E72",
                "E77",
                "E78",
                "E79",
                "E80",
                "E85",
                "E86",
                "E87",
                "E106",
            ],
        }

    elif equipment == "egi128":
        # EGI 128-channel system ROI definitions
        icm_rois = {
            # Scalp ROI - all EEG channels for 128-channel system
            "scalp": [
                f"E{i}"
                for i in range(1, 129)
                if i not in [17, 125, 126, 127, 128]
            ],
            # CNV ROI - central and frontal-central electrodes
            "cnv": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E15",
                "E16",
                "E18",
                "E19",
                "E20",
                "E22",
                "E23",
                "E24",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
            ],
            # MMN ROI - fronto-central electrodes
            "mmn": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E15",
                "E16",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
            ],
            # P3a ROI - fronto-central electrodes
            "p3a": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E103",
                "E104",
                "E105",
                "E106",
            ],
            # P3b ROI - centro-parietal electrodes
            "p3b": [
                "E7",
                "E31",
                "E37",
                "E40",
                "E41",
                "E42",
                "E47",
                "E53",
                "E54",
                "E55",
                "E60",
                "E61",
                "E62",
                "E72",
                "E77",
                "E78",
                "E79",
                "E106",
            ],
        }

    else:
        # Standard 10-20 system ICM ROI approximations
        icm_rois = {
            # Scalp ROI - all standard electrodes (updated to match actual 64-channel data)
            "scalp": [
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "P9",
                "P10",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Iz",
                "Oz",
                "O2",
            ],
            # CNV ROI - central and frontal-central region (updated for 64-channel)
            "cnv": ["Fz", "FC1", "FCz", "FC2", "Cz", "C1", "C3", "C2", "C4"],
            # MMN ROI - fronto-central region for mismatch negativity (updated for 64-channel)
            "mmn": [
                "Fz",
                "F1",
                "F3",
                "F2",
                "F4",
                "FC1",
                "FC3",
                "FCz",
                "FC2",
                "FC4",
                "FC5",
                "FC6",
            ],
            # P3a ROI - fronto-central region (updated for 64-channel)
            "p3a": ["Fz", "F1", "F3", "F2", "F4", "FC1", "FCz", "FC2", "Cz"],
            # P3b ROI - centro-parietal region (updated for 64-channel)
            "p3b": [
                "Cz",
                "CP1",
                "CP3",
                "CPz",
                "CP2",
                "CP4",
                "Pz",
                "P1",
                "P3",
                "P2",
                "P4",
            ],
        }

    return icm_rois


def get_data_for_rois(
    data: np.ndarray,
    ch_names: List[str],
    rois: List[str],
    equipment: str = "standard",
) -> Dict[str, np.ndarray]:
    """Extract data for specified ROIs.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_channels, ...).
    ch_names : list of str
        Channel names corresponding to first dimension of data.
    rois : list of str
        ROI names to extract.
    equipment : str
        Equipment configuration for ICM-specific ROIs: 'standard', 'egi256', or 'egi128'.

    Returns
    -------
    dict
        Dictionary mapping ROI names to data arrays.
    """
    # Get both standard and ICM-specific ROI mappings
    standard_roi_mapping = get_roi_mapping()
    icm_roi_mapping = get_icm_roi_mapping(equipment)

    # Combine mappings (ICM ROIs take precedence)
    roi_mapping = {**standard_roi_mapping, **icm_roi_mapping}

    roi_data = {}

    for roi in rois:
        if roi not in roi_mapping:
            # If ROI not in mapping, treat as individual electrode
            if roi in ch_names:
                roi_idx = ch_names.index(roi)
                roi_data[roi] = data[roi_idx : roi_idx + 1]  # Keep 2D
            else:
                raise ValueError(
                    f"ROI '{roi}' not found in channel names or ROI mapping for equipment '{equipment}'",
                )
        else:
            # Get electrodes for this ROI
            roi_electrodes = roi_mapping[roi]
            roi_indices = []

            for electrode in roi_electrodes:
                if electrode in ch_names:
                    roi_indices.append(ch_names.index(electrode))

            if not roi_indices:
                # Attempt fallback to other equipment configurations
                for alt_equipment in ("egi256", "egi128"):
                    if alt_equipment == equipment:
                        continue
                    alt_mapping = get_icm_roi_mapping(alt_equipment)
                    if roi in alt_mapping:
                        for electrode in alt_mapping[roi]:
                            if electrode in ch_names:
                                roi_indices.append(ch_names.index(electrode))
                        if roi_indices:
                            break

            if not roi_indices:
                raise ValueError(
                    f"No electrodes found for ROI '{roi}' in channel names for any supported equipment (tried '{equipment}', 'egi256', 'egi128')",
                )

            roi_data[roi] = data[roi_indices]

    return roi_data


def aggregate_data(
    data: np.ndarray,
    method: str,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Aggregate data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Data to aggregate.
    method : str
        Aggregation method: 'mean', 'std', 'median', 'min', 'max', 'trim_mean80'.
    axis : int, optional
        Axis along which to aggregate. If None, aggregate over all.

    Returns
    -------
    np.ndarray
        Aggregated data.
    """
    if method == "mean":
        return np.mean(data, axis=axis)
    if method == "std":
        return np.std(data, axis=axis)
    if method == "median":
        return np.median(data, axis=axis)
    if method == "min":
        return np.min(data, axis=axis)
    if method == "max":
        return np.max(data, axis=axis)
    if method == "trim_mean80":
        # NICE-compatible trimmed mean: remove top/bottom 10% (use 80% of data)
        return stats.trim_mean(data, proportiontocut=0.1, axis=axis)
    raise ValueError(f"Unknown aggregation method: {method}")


def _add_meta_to_result(
    result_dict: Dict[str, Any], meta: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Add meta field to result dictionary if provided."""
    if meta is not None:
        result_dict["meta"] = meta
    return result_dict


def apply_roi_trial_aggregation(
    data: Dict[str, np.ndarray],
    roi_aggregation_methods: Optional[List[str]] = None,
    trial_aggregation_methods: Optional[List[str]] = None,
    marker_name: str = "marker",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Apply ROI and trial aggregation to marker data.

    Parameters
    ----------
    data : dict
        Dictionary mapping ROI names to data arrays of shape (n_electrodes, n_trials).
    roi_aggregation_methods : list of str, optional
        Methods to aggregate across ROI electrodes.
    trial_aggregation_methods : list of str, optional
        Methods to aggregate across trials.
    marker_name : str
        Base name for the marker.

    Returns
    -------
    dict
        Dictionary with aggregated results in junifer format.
    """
    results = {}

    # Use a single base output name that junifer recognizes
    base_output_name = (
        marker_name.lower().replace("spectralpower_", "").replace("_", "")
    )

    # Default aggregation if none specified
    if roi_aggregation_methods is None and trial_aggregation_methods is None:
        # Return per-electrode, per-trial data - NO AGGREGATION
        all_values = []
        col_names = []

        for roi_name, roi_data in data.items():
            if roi_data.ndim == 1:
                # Single trial case
                for i in range(len(roi_data)):
                    all_values.append(roi_data[i])
                    col_names.append(f"{roi_name}_elec_{i}")
            else:
                # Multiple trials - keep all trials, don't average
                n_electrodes, n_trials = roi_data.shape
                # Flatten to (n_electrodes * n_trials,) to preserve all trial data
                flattened_data = roi_data.flatten(
                    order="F"
                )  # Column-major order (trials vary fastest)
                for trial_idx in range(n_trials):
                    for elec_idx in range(n_electrodes):
                        idx = trial_idx * n_electrodes + elec_idx
                        all_values.append(flattened_data[idx])
                        col_names.append(
                            f"{roi_name}_trial_{trial_idx}_elec_{elec_idx}"
                        )

        # Handle data reshaping based on marker type and data structure
        if len(all_values) > 0:
            # Determine structure from first ROI
            first_roi_data = next(iter(data.values()))
            if first_roi_data.ndim > 1 and marker_name in [
                "cnvslope",
                "cnvintercept",
            ]:
                # CNV-specific: reshape to (n_trials, n_electrodes) to preserve trial structure
                n_electrodes, n_trials = first_roi_data.shape
                result_data = np.array(all_values).reshape(
                    n_trials, n_electrodes
                )
            else:
                # Default behavior for other markers: flatten to (1, n_features)
                result_data = np.array(all_values).reshape(1, -1)
        else:
            result_data = np.array([]).reshape(0, 0)

        result_dict = {
            "data": result_data,
            "col_names": col_names,
        }
        if meta is not None:
            result_dict["meta"] = meta
        results[base_output_name] = result_dict
        return results

    # Handle case with no ROI aggregation but trial aggregation
    if roi_aggregation_methods is None:
        for trial_agg in trial_aggregation_methods:
            all_values = []
            col_names = []

            for roi_name, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial
                    roi_values = roi_data
                else:
                    # Multiple trials
                    roi_values = aggregate_data(roi_data, trial_agg, axis=1)

                # Each electrode in ROI gets its own column
                for i in range(len(roi_values)):
                    col_names.append(f"{roi_name}_trial_{trial_agg}_elec_{i}")
                    all_values.append(roi_values[i])

            results[base_output_name] = {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            }
        return results

    # Handle case with no trial aggregation but ROI aggregation
    if trial_aggregation_methods is None:
        for roi_agg in roi_aggregation_methods:
            all_values = []
            col_names = []

            for roi_name, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial
                    roi_value = aggregate_data(roi_data, roi_agg)
                    col_names.append(f"{roi_name}_roi_{roi_agg}")
                    all_values.append(roi_value)
                else:
                    # Multiple trials - aggregate across ROI for each trial separately
                    for trial_idx in range(roi_data.shape[1]):
                        trial_data = roi_data[
                            :, trial_idx
                        ]  # Shape: (n_channels,)
                        roi_value = aggregate_data(trial_data, roi_agg)
                        col_names.append(
                            f"{roi_name}_trial_{trial_idx}_roi_{roi_agg}"
                        )
                        all_values.append(roi_value)

            results[base_output_name] = {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            }
        return results

    # Full case with both ROI and trial aggregation
    all_values = []
    col_names = []

    for trial_agg in trial_aggregation_methods:
        for roi_agg in roi_aggregation_methods:
            # Combine all ROIs/channels into one for aggregation
            all_roi_data = []

            for _, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial - add to combined data
                    all_roi_data.append(roi_data)
                else:
                    # Multiple trials - add to combined data
                    all_roi_data.append(roi_data)

            # Concatenate all ROI data along channel axis
            if all_roi_data[0].ndim == 1:
                # Single trial case
                combined_data = np.concatenate(
                    all_roi_data, axis=0
                )  # Shape: (all_channels,)
                roi_value = aggregate_data(combined_data, roi_agg)
            else:
                # Multiple trials case
                combined_data = np.concatenate(
                    all_roi_data, axis=0
                )  # Shape: (all_channels, n_trials)
                # IMPORTANT: Apply ROI aggregation FIRST, then trial aggregation
                # This matches NICE's order: channels first, then epochs
                roi_aggregated = aggregate_data(
                    combined_data, roi_agg, axis=0
                )  # Shape: (n_trials,) - mean across channels
                roi_value = aggregate_data(
                    roi_aggregated, trial_agg
                )  # Shape: scalar - trim_mean80 across trials

            col_names.append(f"all_channels_trial_{trial_agg}_roi_{roi_agg}")
            all_values.append(roi_value)

    result_dict = {
        "data": np.array(all_values).reshape(1, -1),
        "col_names": col_names,
    }
    results[base_output_name] = _add_meta_to_result(result_dict, meta)

    return results
