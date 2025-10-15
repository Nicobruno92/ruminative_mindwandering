#!/usr/bin/env python3
"""
Create PKL file from H5 markers and FIF metadata.

This script reads:
1. Metadata from .fif file (epochs, channels, annotations)
2. Marker values from .h5 file (computed features)
3. Generates .pkl file with exact structure matching the reference

Usage:
    python create_pkl_from_h5_fif.py <h5_file> <fif_file> <output_pkl>
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import mne
import numpy as np
import pandas as pd

from junifer_hdf5_reader_final import JuniferHDF5Reader


def read_fif_metadata(fif_path: str) -> Dict[str, Any]:
    """Read metadata from .fif file."""
    print(f"Reading metadata from {fif_path}")

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

    # Filter to EEG channels only (exclude stimulus channels) to match original pkl
    eeg_picks = mne.pick_types(
        epochs.info, eeg=True, stim=False, exclude="bads"
    )
    epochs = epochs.pick(eeg_picks)

    # Extract epoch annotations
    epoch_annotations = []
    events = epochs.events
    event_id = epochs.event_id
    code_to_desc = {code: desc for desc, code in event_id.items()}

    for i, event in enumerate(events):
        event_time, _, event_code = event
        event_desc = code_to_desc.get(event_code, f"Unknown_{event_code}")

        annotation = {
            "epoch_index": i,
            "epoch_id": f"epoch_{i:04d}",
            "event_time": event_time,
            "event_code": event_code,
            "event_description": event_desc,
            "original_description": event_desc,
        }

        # Parse behavioral parameters from description if structured
        if isinstance(event_desc, str) and "/" in event_desc:
            parts = event_desc.split("/")
            for part in parts:
                part = part.strip()
                if part in ["go", "nogo"]:
                    annotation["response_type"] = part
                elif part in ["correct", "incorrect"]:
                    annotation["accuracy"] = part
                elif part.startswith("valence") and len(part) > 7:
                    try:
                        annotation["valence"] = int(part[7:])
                    except ValueError:
                        pass
                elif part.startswith("confidence") and len(part) > 10:
                    try:
                        annotation["confidence"] = int(part[10:])
                    except ValueError:
                        pass
                elif part.startswith("time") and len(part) > 4:
                    try:
                        annotation["reaction_time"] = int(part[4:])
                    except ValueError:
                        pass
                elif part.startswith("probe") and len(part) > 5:
                    try:
                        annotation["probe_id"] = int(part[5:])
                    except ValueError:
                        pass
                elif part.startswith("onoff") and len(part) > 5:
                    try:
                        annotation["onoff"] = int(part[5:])
                    except ValueError:
                        pass
                elif part.startswith("selfother") and len(part) > 9:
                    try:
                        annotation["selfother"] = int(part[9:])
                    except ValueError:
                        pass
                elif part.startswith("average") and len(part) > 7:
                    try:
                        annotation["average"] = int(part[7:])
                    except ValueError:
                        pass
                elif part.isdigit():
                    annotation["trial_id"] = int(part)

        # Add missing fields with nan
        for field in [
            "response_type",
            "accuracy",
            "valence",
            "confidence",
            "reaction_time",
            "probe_id",
            "trial_id",
            "onoff",
            "selfother",
            "average",
        ]:
            if field not in annotation:
                annotation[field] = np.nan

        epoch_annotations.append(annotation)

    metadata = {
        "epochs": epochs,
        "epoch_annotations": epoch_annotations,
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "channel_names": epochs.ch_names,
        "sfreq": epochs.info["sfreq"],
        "tmin": epochs.tmin,
        "tmax": epochs.tmax,
    }

    print(
        f"Extracted metadata: {len(epochs)} epochs, {len(epochs.ch_names)} channels"
    )
    return metadata


def read_h5_markers(h5_path: str, element: dict) -> Dict[str, Any]:
    """Read marker data from H5 file for a specific element.
    
    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file (per-element file)
    element : dict
        Element information with keys: 'subject', 'task', 'desc'
        (Not used for per-element files, kept for compatibility)
    
    Returns
    -------
    dict
        Dictionary of marker data
    """
    print(f"Reading markers from {h5_path}")
    print(f"  Element: {element}")

    reader = JuniferHDF5Reader(h5_path)
    features = reader.list_features()

    markers_data = {}
    for feature in features:
        print(f"  Reading feature: {feature}")
        try:
            # Read feature data directly
            # Per-element H5 files contain only data for one element
            feature_data = reader.storage.read(feature_name=feature)
            
            # Store the feature data as-is
            # Per-element files don't need element matching
            markers_data[feature] = feature_data
            print(f"    ✓ Successfully read {feature}")
                
        except Exception as e:
            print(f"    Warning: Could not read {feature}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Read {len(markers_data)} features from H5 file")
    return markers_data


def create_spectral_marker(
    h5_data: Dict,
    marker_name: str,
    channel_names: List[str],
    epoch_annotations: List[Dict],
) -> Dict[str, Any]:
    """Create spectral marker structure from H5 data."""

    # Find the spectral feature in H5 data
    spectral_feature = None
    for feature_name, feature_data in h5_data.items():
        if (
            "spectralpower" in feature_name.lower()
            and marker_name in feature_name.lower()
        ):
            spectral_feature = feature_data
            break

    if spectral_feature is None:
        print(f"Warning: No spectral feature found for {marker_name}")
        return None

    data = spectral_feature["data"]
    headers = spectral_feature["column_headers"]

    # Parse headers: format is "band_channel_epoch_XXXX"
    marker_data = {}

    for i, header in enumerate(headers):
        parts = header.split("_")
        if len(parts) >= 4:
            band = parts[0]
            channel = parts[1]
            epoch = parts[2] + "_" + parts[3]  # "epoch_XXXX"

            if channel not in marker_data:
                marker_data[channel] = {}

            band_epoch_key = f"{band}_{epoch}"
            marker_data[channel][band_epoch_key] = float(data[i, 0])

    return {
        "access_pattern": "channel_epoch",
        "channel_names": channel_names,
        "epoch_annotations": epoch_annotations,
        "data": marker_data,
    }


def create_erp_marker(
    h5_data: Dict,
    marker_name: str,
    channel_names: List[str],
    epoch_annotations: List[Dict],
) -> Dict[str, Any]:
    """Create ERP marker structure from H5 data.
    
    This handles TimeLockedTopography, PermutationEntropy, and KolmogorovComplexity markers.
    """

    # Find the feature in H5 data - it should be the only one in h5_data dict
    if not h5_data:
        print(f"Warning: No data provided for {marker_name}")
        return None
    
    # Get the first (and only) feature from h5_data
    feature_name = list(h5_data.keys())[0]
    erp_feature = h5_data[feature_name]
    
    print(f"    Creating ERP-like marker from feature: {feature_name}")

    data = erp_feature["data"]
    headers = erp_feature["column_headers"]

    # For ERP-like markers: headers are channel names, data is organized as epochs × channels
    marker_data = {}

    # Create channel mapping - headers might have suffixes like '_ch0'
    hdf5_channels = []
    for header in headers:
        # Remove common suffixes
        channel = header.replace("_ch0", "").replace("_0", "")
        hdf5_channels.append(channel)

    n_epochs = len(epoch_annotations)
    n_channels = len(hdf5_channels)

    print(f"    Data shape: {data.shape}, Headers: {len(headers)}, Epochs: {n_epochs}, Channels: {n_channels}")

    # HDF5 data is organized as: [epoch0_ch0, epoch0_ch1, ..., epoch1_ch0, ...]
    for epoch_idx in range(n_epochs):
        epoch_key = f"epoch_{epoch_idx:04d}"

        for ch_idx, channel in enumerate(hdf5_channels):
            if channel in channel_names:
                hdf5_idx = epoch_idx * n_channels + ch_idx

                if hdf5_idx < len(data):
                    if channel not in marker_data:
                        marker_data[channel] = {}

                    marker_data[channel][epoch_key] = float(data[hdf5_idx, 0])

    print(f"    Created marker data for {len(marker_data)} channels")

    return {
        "access_pattern": "channel_epoch",
        "channel_names": channel_names,
        "epoch_annotations": epoch_annotations,
        "data": marker_data,
    }


def create_connectivity_marker(
    h5_data: Dict,
    marker_name: str,
    channel_names: List[str],
    epoch_annotations: List[Dict],
) -> Dict[str, Any]:
    """Create connectivity marker structure from H5 data."""

    # Find the connectivity feature in H5 data
    conn_feature = None
    for feature_name, feature_data in h5_data.items():
        if marker_name.lower() in feature_name.lower():
            conn_feature = feature_data
            break

    if conn_feature is None:
        print(f"Warning: No connectivity feature found for {marker_name}")
        return None

    data = conn_feature["data"]

    # Connectivity data is typically organized as epochs × channel_pairs
    n_epochs = len(epoch_annotations)
    n_pairs = data.shape[1] if data.ndim > 1 else len(data)
    n_channels = int(np.sqrt(n_pairs))

    marker_data = {}

    # Create channel pairs
    for epoch_idx in range(min(n_epochs, data.shape[0])):
        epoch_key = f"epoch_{epoch_idx:04d}"

        pair_idx = 0
        for i in range(n_channels):
            for j in range(n_channels):
                if i < len(channel_names) and j < len(channel_names):
                    ch1 = channel_names[i]
                    ch2 = channel_names[j]
                    pair_name = f"{ch1}-{ch2}"

                    if pair_name not in marker_data:
                        marker_data[pair_name] = {}

                    if pair_idx < n_pairs:
                        marker_data[pair_name][epoch_key] = float(
                            data[epoch_idx, pair_idx]
                        )

                    pair_idx += 1

    return {
        "access_pattern": "channel_pair_epoch",
        "channel_names": channel_names,
        "epoch_annotations": epoch_annotations,
        "data": marker_data,
    }


def create_pkl_from_h5_fif(h5_path: str, fif_path: str, output_path: str, 
                           element: dict = None):
    """Main function to create PKL file from H5 and FIF files.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    fif_path : str
        Path to FIF file
    output_path : str
        Path for output PKL file
    element : dict, optional
        Element information dict with keys: 'subject', 'task', 'desc'
        Required when reading from multi-element H5 files
    """

    print("Creating PKL file from:")
    print(f"  H5 file: {h5_path}")
    print(f"  FIF file: {fif_path}")
    print(f"  Output: {output_path}")
    if element:
        print(f"  Element: {element}")

    # Read metadata from FIF
    metadata = read_fif_metadata(fif_path)

    # Read markers from H5 (with element info if provided)
    h5_markers = read_h5_markers(h5_path, element) if element else read_h5_markers(h5_path, {})

    # Create PKL structure
    pkl_data = {
        "markers": {},
        "metadata": {},
        "annotations": {"epochs": metadata["epoch_annotations"]},
        "info": {
            "n_epochs": metadata["n_epochs"],
            "n_channels": metadata["n_channels"],
            "channel_names": metadata["channel_names"],
            "created_at": str(pd.Timestamp.now()),
            "storage_type": "pickle",
            "pipeline_config": "generated_from_h5_fif",
            "n_markers": 0,
            "access_patterns": {},
        },
    }

    # Create markers directly from H5 features
    # Feature names from H5 are like: EEG_psd_bands_spectralpower
    # We extract the marker name and determine type automatically
    
    for feature_name, feature_data in h5_markers.items():
        print(f"Processing H5 feature: {feature_name}")
        
        # Extract marker name from feature name
        # Pattern: EEG_{marker_name}_{kind}
        parts = feature_name.split('_')
        if len(parts) < 3:
            print(f"  Skipping: unexpected feature name format")
            continue
            
        # Remove 'EEG' prefix and marker kind suffix
        marker_name = '_'.join(parts[1:-1])  # e.g., 'psd_bands', 'wsmi_theta', 'P1'
        marker_kind = parts[-1].lower()  # e.g., 'spectralpower', 'timelockedtopo'
        
        print(f"  Marker name: {marker_name}, kind: {marker_kind}")
        
        # Determine marker type based on kind
        if 'spectral' in marker_kind:
            marker_type = 'spectral'
        elif 'timelockedtopo' in marker_kind:
            marker_type = 'erp'
        elif 'permutationentropy' in marker_kind or 'kolmogorov' in marker_kind:
            marker_type = 'erp'  # These use same structure as ERP
        elif 'symbolicmutualinformation' in marker_kind:
            marker_type = 'connectivity'
        else:
            print(f"  Warning: Unknown marker kind: {marker_kind}, skipping")
            continue
        
        # Create marker based on type
        marker = None
        if marker_type == "spectral":
            marker = create_spectral_marker(
                {feature_name: feature_data},  # Pass as dict with feature name as key
                marker_name,
                metadata["channel_names"],
                metadata["epoch_annotations"],
            )
        elif marker_type == "erp":
            marker = create_erp_marker(
                {feature_name: feature_data},
                marker_name,
                metadata["channel_names"],
                metadata["epoch_annotations"],
            )
        elif marker_type == "connectivity":
            marker = create_connectivity_marker(
                {feature_name: feature_data},
                marker_name,
                metadata["channel_names"],
                metadata["epoch_annotations"],
            )

        if marker is not None:
            pkl_data["markers"][marker_name] = marker
            pkl_data["info"]["access_patterns"][marker_name] = marker[
                "access_pattern"
            ]
            print(f"  ✓ Created marker: {marker_name}")
        else:
            print(f"  ✗ Failed to create marker: {marker_name}")

    # Update marker count
    pkl_data["info"]["n_markers"] = len(pkl_data["markers"])

    # Save PKL file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Successfully created PKL file: {output_path}")
    print(f"Generated {len(pkl_data['markers'])} markers:")
    for name, pattern in pkl_data["info"]["access_patterns"].items():
        print(f"  {name}: {pattern}")


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create PKL file from H5 markers and FIF metadata"
    )
    parser.add_argument("h5_file", help="Path to HDF5 file")
    parser.add_argument("fif_file", help="Path to FIF file")
    parser.add_argument("output_pkl", help="Path for output PKL file")
    parser.add_argument("--subject", help="Subject ID (e.g., sub-31)")
    parser.add_argument("--task", help="Task name (e.g., Sart4)")
    parser.add_argument("--desc", help="Description (e.g., evoked, state)")
    
    args = parser.parse_args()
    
    # Construct element dict if all fields provided
    element = None
    if args.subject and args.task and args.desc:
        element = {
            'subject': args.subject,
            'task': args.task,
            'desc': args.desc
        }
        print(f"Using element: {element}")

    create_pkl_from_h5_fif(args.h5_file, args.fif_file, args.output_pkl, element=element)


if __name__ == "__main__":
    main()
