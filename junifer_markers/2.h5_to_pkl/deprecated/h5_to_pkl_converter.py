#!/usr/bin/env python3
"""
H5 to PKL Converter with FIF Metadata Integration

This converter reads marker results from Junifer HDF5 files and combines them
with epoch metadata from MNE FIF files to create a comprehensive PKL file
with hierarchical indexing: markers -> epochs -> channels with full metadata.

Usage:
    python h5_to_pkl_converter.py <fif_file> <h5_file> <output_pkl>
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mne
import numpy as np

try:
    from junifer.storage import HDF5FeatureStorage
except ImportError:
    print("Warning: junifer not found. HDF5 reading may not work.")

logger = logging.getLogger(__name__)

# EXPLICIT CATALOG OF ALL EXPECTED MARKERS - NO INFERENCES ALLOWED
EXPECTED = {
    "EEG_psd_bands_spectralpower":        ("spectral",   lambda n_ep, n_ch: (n_ep, 64, 5)),
    "EEG_psd_relative_spectralpower":     ("spectral",   lambda n_ep, n_ch: (n_ep, 64, 5)),
    "EEG_wsmi_theta_symbolicmutualinformation": ("connectivity", lambda n_ep, n_ch: (n_ep, 2016)),
    "EEG_wsmi_alpha_symbolicmutualinformation": ("connectivity", lambda n_ep, n_ch: (n_ep, 2016)),
    "EEG_wsmi_beta_symbolicmutualinformation":  ("connectivity", lambda n_ep, n_ch: (n_ep, 2016)),
    "EEG_wsmi_gamma_symbolicmutualinformation": ("connectivity", lambda n_ep, n_ch: (n_ep, 2016)),
    "EEG_kolmogorov_complexity_kolmogorovcomplexity": ("information_theory", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_PE_theta_permutationentropy":    ("information_theory", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_PE_alpha_permutationentropy":    ("information_theory", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_PE_beta_permutationentropy":     ("information_theory", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_PE_gamma_permutationentropy":    ("information_theory", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_P1_timelockedtopo":              ("time_locked", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_N1_timelockedtopo":              ("time_locked", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_P2_timelockedtopo":              ("time_locked", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_P3a_timelockedtopo":             ("time_locked", lambda n_ep, n_ch: (n_ep, n_ch)),
    "EEG_P3b_timelockedtopo":             ("time_locked", lambda n_ep, n_ch: (n_ep, n_ch)),
}


class EpochMetadata:
    """Container for epoch-level metadata."""

    def __init__(
        self,
        epoch_idx: int,
        event: np.ndarray,
        annotation: Optional[mne.Annotations] = None,
        behavioral_data: Optional[Dict] = None,
    ):
        self.epoch_idx = epoch_idx
        self.event = event  # [sample, prev_id, event_id]
        self.event_id = event[2] if len(event) > 2 else None
        self.onset_sample = event[0] if len(event) > 0 else None
        self.annotation = annotation
        self.behavioral_data = behavioral_data or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch_idx": self.epoch_idx,
            "event": self.event.tolist()
            if isinstance(self.event, np.ndarray)
            else self.event,
            "event_id": self.event_id,
            "onset_sample": self.onset_sample,
            "annotation": {
                "onset": self.annotation.onset[0]
                if self.annotation and len(self.annotation.onset) > 0
                else None,
                "duration": self.annotation.duration[0]
                if self.annotation and len(self.annotation.duration) > 0
                else None,
                "description": self.annotation.description[0]
                if self.annotation and len(self.annotation.description) > 0
                else None,
            }
            if self.annotation
            else None,
            "behavioral_data": self.behavioral_data,
        }


class ChannelData:
    """Container for channel-level data with metadata access."""

    def __init__(
        self,
        channel_name: str,
        channel_idx: int,
        data: np.ndarray,
        metadata: EpochMetadata,
    ):
        self.channel_name = channel_name
        self.channel_idx = channel_idx
        self.data = data
        self.metadata = metadata

    def __repr__(self):
        return f"ChannelData(ch={self.channel_name}, data_shape={self.data.shape}, event_id={self.metadata.event_id})"


class EpochData:
    """Container for epoch-level data with channel indexing."""

    def __init__(
        self, epoch_idx: int, metadata: EpochMetadata, channel_names: List[str]
    ):
        self.epoch_idx = epoch_idx
        self.metadata = metadata
        self.channel_names = channel_names
        self._channel_data = {}
        # Add annotations as a special accessible key
        self.annotations = metadata.annotation

    def add_channel_data(
        self, channel_name: str, channel_idx: int, data: np.ndarray
    ):
        """Add data for a specific channel."""
        self._channel_data[channel_name] = ChannelData(
            channel_name, channel_idx, data, self.metadata
        )

    def __getitem__(self, channel_name: str) -> ChannelData:
        """Access channel data by name."""
        if channel_name not in self._channel_data:
            raise KeyError(
                f"Channel '{channel_name}' not found. Available: {list(self._channel_data.keys())}"
            )
        return self._channel_data[channel_name]

    def keys(self):
        """Get available channel names."""
        return self._channel_data.keys()

    def items(self):
        """Iterate over channel name, data pairs."""
        return self._channel_data.items()

    def __repr__(self):
        return f"EpochData(idx={self.epoch_idx}, event_id={self.metadata.event_id}, channels={len(self._channel_data)})"


class MarkerData:
    """Container for marker-level data with epoch indexing."""

    def __init__(
        self,
        marker_name: str,
        marker_type: str,
        channel_names: List[str],
        n_epochs: int,
    ):
        self.marker_name = marker_name
        self.marker_type = marker_type
        self.channel_names = channel_names
        self.n_epochs = n_epochs
        self._epoch_data = {}
        self.metadata = {}

    def add_epoch_data(
        self, epoch_idx: int, epoch_metadata: EpochMetadata, data: np.ndarray
    ):
        """Add data for a specific epoch."""
        epoch_data = EpochData(epoch_idx, epoch_metadata, self.channel_names)

        # Handle different data shapes based on marker type and reshaped data
        if (
            data.ndim == 1
        ):  # Flattened data (connectivity matrices or single values)
            if self.marker_type in ["connectivity", "wsmi", "smi"]:
                # Connectivity data: create channel pair names for upper triangular matrix
                # Assuming 64x64 connectivity matrix (2016 connections)
                n_channels_conn = 64  # EEG channels used for connectivity
                conn_idx = 0
                for i in range(n_channels_conn):
                    for j in range(i + 1, n_channels_conn):
                        if conn_idx < len(data):
                            ch_i_name = (
                                self.channel_names[i]
                                if i < len(self.channel_names)
                                else f"Ch{i}"
                            )
                            ch_j_name = (
                                self.channel_names[j]
                                if j < len(self.channel_names)
                                else f"Ch{j}"
                            )
                            pair_name = f"{ch_i_name}-{ch_j_name}"
                            epoch_data.add_channel_data(
                                pair_name, conn_idx, np.array([data[conn_idx]])
                            )
                            conn_idx += 1
            else:
                # Single values: one value per channel
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < len(data):
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.array([data[ch_idx]])
                        )
                    else:
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.array([0.0])
                        )
        elif data.ndim == 2:  # Channel x features (spectral or single values)
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < data.shape[0]:
                    epoch_data.add_channel_data(
                        ch_name, ch_idx, data[ch_idx, :]
                    )
                else:
                    # Handle case where data has fewer channels than FIF
                    epoch_data.add_channel_data(
                        ch_name, ch_idx, np.zeros(data.shape[1])
                    )
        elif (
            data.ndim == 3
        ):  # Channel x channel x features (connectivity matrices)
            if self.marker_type in ["connectivity", "wsmi", "smi"]:
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[0]:
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, data[ch_idx, :, :]
                        )
                    else:
                        # Handle missing channels
                        epoch_data.add_channel_data(
                            ch_name,
                            ch_idx,
                            np.zeros((data.shape[1], data.shape[2])),
                        )
        elif (
            data.ndim == 4
        ):  # Channel x channel x freq x time (4D connectivity)
            if self.marker_type in ["connectivity", "wsmi", "smi"]:
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[0]:
                        # Store 3D connectivity data for this channel
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, data[ch_idx, :, :, :]
                        )

        self._epoch_data[epoch_idx] = epoch_data

    def __getitem__(self, epoch_idx: int) -> EpochData:
        """Access epoch data by index."""
        if epoch_idx not in self._epoch_data:
            raise KeyError(
                f"Epoch {epoch_idx} not found. Available: {list(self._epoch_data.keys())}"
            )
        return self._epoch_data[epoch_idx]

    def keys(self):
        """Get available epoch indices."""
        return self._epoch_data.keys()

    def items(self):
        """Iterate over epoch index, data pairs."""
        return self._epoch_data.items()

    def __repr__(self):
        return f"MarkerData(name={self.marker_name}, type={self.marker_type}, epochs={len(self._epoch_data)})"


class H5ToPklConverter:
    """Convert Junifer HDF5 + MNE FIF to organized PKL format."""

    def __init__(self, fif_path: Union[str, Path], h5_path: Union[str, Path]):
        self.fif_path = Path(fif_path)
        self.h5_path = Path(h5_path)

        # Validate files exist
        if not self.fif_path.exists():
            raise FileNotFoundError(f"FIF file not found: {self.fif_path}")
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # Load FIF data
        logger.info(f"Loading FIF file: {self.fif_path}")
        self.epochs = mne.read_epochs(str(self.fif_path), verbose=False)

        # Load HDF5 data
        logger.info(f"Loading HDF5 file: {self.h5_path}")
        self.storage = HDF5FeatureStorage(str(self.h5_path))

        # Extract metadata
        self._extract_epoch_metadata()

    def _extract_epoch_metadata(self):
        """Extract metadata for each epoch from FIF file."""
        self.epoch_metadata = {}

        # Get events and annotations
        events = self.epochs.events
        annotations = self.epochs.annotations

        for epoch_idx in range(len(events)):
            event = events[epoch_idx]

            # Find matching annotation (if any)
            epoch_annotation = None
            if annotations is not None:
                # Find annotation closest to this epoch's onset
                epoch_onset = (
                    event[0] / self.epochs.info["sfreq"]
                )  # Convert to seconds
                for ann_idx in range(len(annotations)):
                    ann_onset = annotations.onset[ann_idx]
                    if abs(ann_onset - epoch_onset) < 0.1:  # Within 100ms
                        epoch_annotation = mne.Annotations(
                            onset=[annotations.onset[ann_idx]],
                            duration=[annotations.duration[ann_idx]],
                            description=[annotations.description[ann_idx]],
                        )
                        break

            self.epoch_metadata[epoch_idx] = EpochMetadata(
                epoch_idx=epoch_idx, event=event, annotation=epoch_annotation
            )

    def _get_marker_type(self, marker_name: str) -> str:
        """Get marker type from explicit catalog - NO INFERENCES."""
        if marker_name not in EXPECTED:
            raise ValueError(f"Unknown marker '{marker_name}' — add it to EXPECTED")
        return EXPECTED[marker_name][0]

    def _get_actual_channel_names(self, marker_type: str) -> List[str]:
        """Get channel names deterministically based on marker type - NO FALLBACKS."""
        if marker_type == "spectral":
            # Spectral markers require exactly 64 EEG channels
            eeg_channels = [
                ch
                for ch in self.epochs.ch_names
                if not ch.startswith("EOG") and ch not in ["VEOG", "HEOG"]
            ]
            if len(eeg_channels) < 64:
                raise ValueError("Spectral markers require exactly 64 EEG channel names")
            return eeg_channels[:64]
        else:
            # All other markers use all FIF channels
            return self.epochs.ch_names

    def _reshape_marker_data(self, marker_name: str, data: np.ndarray) -> np.ndarray:
        """Reshape marker data to expected shape - NO HEURISTICS."""
        if marker_name not in EXPECTED:
            raise ValueError(f"Unknown marker '{marker_name}' — add it to EXPECTED")
        
        marker_type, exp_fn = EXPECTED[marker_name]
        exp_shape = exp_fn(len(self.epochs), len(self.epochs.ch_names))
        
        logger.info(f"Reshaping {marker_name}: input shape {data.shape}, expected {exp_shape}")
        
        # Handle flattened data from H5
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.ravel()
        
        # Check if data matches expected shape exactly
        if tuple(data.shape) == exp_shape:
            return data
        
        # Check if data can be reshaped to expected shape
        if data.ndim == 1 and data.size == int(np.prod(exp_shape)):
            return data.reshape(exp_shape)
        
        # Check if data is aggregated (single value) - NOT ALLOWED for per-epoch flow
        if data.size == 1:
            raise ValueError(f"Marker '{marker_name}' has aggregated data (single value) - not valid for per-epoch flow")
        
        raise ValueError(f"{marker_name}: got shape {tuple(data.shape)} != expected {exp_shape}")

    def convert(self) -> Dict[str, Any]:
        """Convert HDF5 + FIF to organized dictionary structure."""
        logger.info("Starting conversion...")

        # Get all features from HDF5
        features = self.storage.list_features()
        logger.info(f"Found {len(features)} features in HDF5")

        # Organize data by markers
        markers = {}
        metadata = {
            "fif_info": {
                "n_epochs": len(self.epochs),
                "n_channels": len(self.epochs.ch_names),
                "channel_names": self.epochs.ch_names,
                "sfreq": self.epochs.info["sfreq"],
                "tmin": self.epochs.tmin,
                "tmax": self.epochs.tmax,
                "event_id": self.epochs.event_id,
            },
            "h5_info": {
                "n_features": len(features),
                "feature_names": [meta["name"] for meta in features.values()],
            },
        }

        # Group features by name to handle duplicates
        features_by_name = {}
        for feature_id, feature_meta in features.items():
            marker_name = feature_meta["name"]
            if marker_name not in features_by_name:
                features_by_name[marker_name] = []
            features_by_name[marker_name].append((feature_id, feature_meta))

        logger.info(f"Found {len(features_by_name)} unique marker names")

        # Process each unique marker name
        for marker_name, feature_list in features_by_name.items():
            logger.info(
                f"Processing marker: {marker_name} ({len(feature_list)} instances)"
            )

            # STRICT DUPLICATE HANDLING - NO FALLBACKS
            if len(feature_list) > 1:
                ids = [fid for fid, _ in feature_list]
                raise ValueError(f"Duplicate feature name '{marker_name}' with MD5s={ids}")

            try:
                feature_id, feature_meta = feature_list[0]
                # Read using MD5 to avoid any name conflicts
                feature_data = self.storage.read(feature_md5=feature_id)

                data = feature_data["data"]

                # Log data info for debugging
                logger.info(f"  Data shape: {data.shape}, dtype: {data.dtype}")
                if data.size > 0:
                    logger.info(
                        f"  Data range: {np.min(data):.6f} to {np.max(data):.6f}"
                    )
                else:
                    logger.warning(f"  *** EMPTY DATA for {marker_name} ***")

                # Get marker type from explicit catalog
                marker_type = self._get_marker_type(marker_name)
                
                # Reshape data to expected shape
                reshaped_data = self._reshape_marker_data(marker_name, data)

                # Get channel names deterministically
                actual_channel_names = self._get_actual_channel_names(marker_type)

                # Create marker data container
                marker_data = MarkerData(
                    marker_name=marker_name,
                    marker_type=marker_type,
                    channel_names=actual_channel_names,
                    n_epochs=len(self.epochs),
                )

                # Store the raw reshaped data directly in metadata for validation
                marker_data.metadata["raw_data"] = reshaped_data

                # Add epoch data - handle different data structures
                if reshaped_data.size == 0:
                    logger.error(
                        f"No data to process for {marker_name} - skipping"
                    )
                    continue

                # Check if we have per-epoch data
                if reshaped_data.ndim >= 2 and reshaped_data.shape[0] == len(
                    self.epochs
                ):
                    # Data is structured as (epochs, ...)
                    for epoch_idx in range(len(self.epochs)):
                        epoch_metadata = self.epoch_metadata[epoch_idx]

                        if reshaped_data.ndim == 2:
                            epoch_data_slice = reshaped_data[epoch_idx, :]
                        elif reshaped_data.ndim == 3:
                            epoch_data_slice = reshaped_data[epoch_idx, :, :]
                        else:
                            epoch_data_slice = reshaped_data[epoch_idx]

                        marker_data.add_epoch_data(
                            epoch_idx, epoch_metadata, epoch_data_slice
                        )
                elif (
                    reshaped_data.ndim == 1
                    and len(reshaped_data) % len(self.epochs) == 0
                ):
                    # Single flattened array - divide by epochs
                    elements_per_epoch = len(reshaped_data) // len(self.epochs)
                    for epoch_idx in range(len(self.epochs)):
                        epoch_metadata = self.epoch_metadata[epoch_idx]
                        epoch_start = epoch_idx * elements_per_epoch
                        epoch_end = (epoch_idx + 1) * elements_per_epoch
                        epoch_data_slice = reshaped_data[epoch_start:epoch_end]

                        marker_data.add_epoch_data(
                            epoch_idx, epoch_metadata, epoch_data_slice
                        )
                else:
                    # Single value or aggregated data - replicate for all epochs
                    logger.info(
                        f"Replicating single/aggregated data for all epochs: {marker_name}"
                    )
                    for epoch_idx in range(len(self.epochs)):
                        epoch_metadata = self.epoch_metadata[epoch_idx]
                        marker_data.add_epoch_data(
                            epoch_idx, epoch_metadata, reshaped_data
                        )

                markers[marker_name] = marker_data

            except Exception as e:
                # FAIL-FAST: propagate all exceptions, no empty markers
                logger.error(f"Failed to process marker {marker_name}: {e}")
                raise

        # AUDIT: Log successful conversion for each marker
        logger.info("=" * 60)
        logger.info("CONVERSION AUDIT - SUCCESSFUL MARKERS")
        logger.info("=" * 60)
        for name, marker in markers.items():
            logger.info(f"[OK] {name}: {marker.marker_type}, {len(marker._epoch_data)} epochs")
        
        # AUDIT: Check for missing expected markers
        logger.info("=" * 60)
        logger.info("CONVERSION AUDIT - MISSING MARKERS")
        logger.info("=" * 60)
        missing_markers = []
        for name, (mtype, exp_fn) in EXPECTED.items():
            if name not in markers:
                missing_markers.append(name)
                logger.warning(f"[MISSING] Expected marker not found: {name}")
        
        if missing_markers:
            logger.warning(f"Total missing expected markers: {len(missing_markers)}")
        else:
            logger.info("All expected markers found successfully!")

        result = {
            "markers": markers,
            "metadata": metadata,
            "epoch_metadata": {
                idx: meta.to_dict()
                for idx, meta in self.epoch_metadata.items()
            },
        }

        logger.info(f"Conversion complete. Processed {len(markers)} markers.")
        return result

    def save_pkl(self, output_path: Union[str, Path], data: Dict[str, Any]):
        """Save converted data to PKL file."""
        output_path = Path(output_path)
        logger.info(f"Saving to PKL: {output_path}")

        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"PKL file saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )


def demonstrate_usage(pkl_data: Dict[str, Any]):
    """Demonstrate how to use the converted PKL data."""
    print("\n" + "=" * 60)
    print("USAGE DEMONSTRATION")
    print("=" * 60)

    markers = pkl_data["markers"]
    metadata = pkl_data["metadata"]

    print(f"\n1. Available markers ({len(markers)}):")
    for marker_name, marker_data in markers.items():
        print(
            f"   - {marker_name} ({marker_data.marker_type}): {len(marker_data.keys())} epochs"
        )

    print("\n2. Metadata:")
    print(f"   - Epochs: {metadata['fif_info']['n_epochs']}")
    print(f"   - Channels: {metadata['fif_info']['n_channels']}")
    print(f"   - Sampling rate: {metadata['fif_info']['sfreq']} Hz")

    if markers:
        # Demonstrate hierarchical access
        first_marker_name = next(iter(markers.keys()))
        first_marker = markers[first_marker_name]

        print(
            f"\n3. Hierarchical access example (marker: {first_marker_name}):"
        )

        if first_marker.keys():
            first_epoch_idx = next(iter(first_marker.keys()))
            first_epoch = first_marker[first_epoch_idx]

            print(f"   - Epoch {first_epoch_idx} metadata:")
            print(f"     Event ID: {first_epoch.metadata.event_id}")
            print(f"     Onset sample: {first_epoch.metadata.onset_sample}")

            if first_epoch.keys():
                first_channel = next(iter(first_epoch.keys()))
                channel_data = first_epoch[first_channel]

                print(f"   - Channel '{first_channel}' data:")
                print(f"     Shape: {channel_data.data.shape}")
                print(
                    f"     Value range: [{np.min(channel_data.data):.6f}, {np.max(channel_data.data):.6f}]"
                )

                print("\n4. Access pattern:")
                print(
                    f"   pkl_data['markers']['{first_marker_name}'][{first_epoch_idx}]['{first_channel}'].data"
                )
                print(
                    f"   pkl_data['markers']['{first_marker_name}'][{first_epoch_idx}]['{first_channel}'].metadata"
                )


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert Junifer HDF5 + MNE FIF to organized PKL"
    )
    parser.add_argument("fif_file", help="Path to MNE epochs FIF file")
    parser.add_argument("h5_file", help="Path to Junifer HDF5 features file")
    parser.add_argument("output_pkl", help="Path for output PKL file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Show usage demonstration"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        # Create converter and run conversion
        converter = H5ToPklConverter(args.fif_file, args.h5_file)
        data = converter.convert()
        converter.save_pkl(args.output_pkl, data)

        if args.demo:
            demonstrate_usage(data)

        print("\nConversion successful!")
        print(f"Input FIF: {args.fif_file}")
        print(f"Input HDF5: {args.h5_file}")
        print(f"Output PKL: {args.output_pkl}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
