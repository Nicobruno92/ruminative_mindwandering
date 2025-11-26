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
import math
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
                channel_count = len(self.channel_names)

                # Case 1: Aggregated connectivity (per-channel values)
                if channel_count > 0 and len(data) == channel_count:
                    for ch_idx, ch_name in enumerate(self.channel_names):
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.array([data[ch_idx]])
                        )
                elif channel_count == 0 and len(data) > 0:
                    # No channel metadata available - synthesize channel names
                    for ch_idx in range(len(data)):
                        ch_name = f"Ch{ch_idx}"
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.array([data[ch_idx]])
                        )
                else:
                    # Case 2: Flattened upper-triangular connectivity pairs
                    if channel_count == 0:
                        # Infer number of channels from number of pairwise connections
                        inferred = max(
                            2,
                            int((1 + math.isqrt(1 + 8 * len(data))) // 2),
                        )
                        channel_names = [f"Ch{i}" for i in range(inferred)]
                    else:
                        channel_names = self.channel_names

                    n_channels_conn = len(channel_names)
                    conn_idx = 0
                    for i in range(n_channels_conn):
                        for j in range(i + 1, n_channels_conn):
                            if conn_idx >= len(data):
                                break
                            ch_i_name = channel_names[i]
                            ch_j_name = channel_names[j]
                            pair_name = f"{ch_i_name}-{ch_j_name}"
                            epoch_data.add_channel_data(
                                pair_name, conn_idx, np.array([data[conn_idx]])
                            )
                            conn_idx += 1
                        if conn_idx >= len(data):
                            break
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
            if self.marker_type in ["sleep_spindles", "sleep_slow_waves"]:
                # Sleep markers: data is (features, channels) after transpose
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[1]:  # channels are in dim 1 for sleep markers
                        # Extract all features for this channel: data[:, ch_idx]
                        channel_features = data[:, ch_idx]  # (n_features,)
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, channel_features
                        )
                    else:
                        # Handle missing channels
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.zeros(data.shape[0])
                        )
            elif "spectral" in self.marker_type.lower() or "psd" in self.marker_type.lower():
                # Spectral power markers: data is (bands, channels) after transpose
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[1]:  # channels are in dim 1 for spectral markers
                        # Extract all frequency bands for this channel: data[:, ch_idx]
                        channel_bands = data[:, ch_idx]  # (n_bands,)
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, channel_bands
                        )
                    else:
                        # Handle missing channels
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.zeros(data.shape[0])
                        )
            else:
                # Original logic for (channels, features) data
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
            logger.info(f"DEBUG: 3D data in add_epoch_data - shape: {data.shape}, marker_type: {self.marker_type}")
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
            elif self.marker_type in ["sleep_spindles", "sleep_slow_waves"]:
                logger.info(f"DEBUG: Processing 3D sleep marker data - shape: {data.shape}")
                # Sleep markers: (epochs, features, channels) after transpose
                # Need to extract data[epoch_idx, :, ch_idx] for each channel
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[2]:  # channels are in dim 2 for sleep markers
                        # Extract all features for this channel: data[epoch_idx, :, ch_idx]
                        channel_features = data[:, :, ch_idx]  # (epochs, features)
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, channel_features
                        )
                    else:
                        # Handle missing channels
                        epoch_data.add_channel_data(
                            ch_name,
                            ch_idx,
                            np.zeros(data.shape[1]),
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

        # Load FIF-derived context
        logger.info(f"Loading FIF file: {self.fif_path}")
        self._load_fif_context()

        # Load HDF5 data
        logger.info(f"Loading HDF5 file: {self.h5_path}")
        self.storage = HDF5FeatureStorage(str(self.h5_path))

        # Extract metadata
        self._extract_epoch_metadata()

    def _load_fif_context(self):
        """Load epochs when possible, otherwise rely on info/events/annotations."""
        self.epochs = None
        self.info = None
        self.events = None
        self.annotations = None

        try:
            self.epochs = mne.read_epochs(str(self.fif_path), verbose=False)
            self.info = self.epochs.info
            self.events = self.epochs.events
            self.annotations = self.epochs.annotations
            self.n_epochs = len(self.epochs)
            self.channel_names = self.epochs.ch_names
            self.sfreq = float(self.epochs.info["sfreq"])
            self.tmin = float(self.epochs.tmin)
            self.tmax = float(self.epochs.tmax)
            self.event_id = self.epochs.event_id
            logger.info("Loaded epochs directly from FIF.")
        except Exception as err:
            logger.warning(
                "Failed to read epochs (%s). Falling back to metadata-only mode.",
                err,
            )
            self.info = mne.io.read_info(str(self.fif_path), verbose=False)
            self.channel_names = self.info["ch_names"]
            self.sfreq = float(self.info["sfreq"])
            try:
                self.events = mne.read_events(str(self.fif_path))
            except Exception as event_err:
                raise RuntimeError(
                    f"Unable to read events from FIF file: {event_err}"
                ) from event_err

            try:
                self.annotations = mne.read_annotations(str(self.fif_path))
            except Exception:
                self.annotations = None

            self.n_epochs = len(self.events)
            self.tmin = None
            self.tmax = None
            self.event_id = self._infer_event_id_from_events(self.events)
            logger.info(
                "Metadata-only mode: %d epochs inferred from events.",
                self.n_epochs,
            )

    def _infer_event_id_from_events(
        self, events: np.ndarray
    ) -> Dict[str, int]:
        """Create a fallback event_id mapping from event codes."""
        event_codes = np.unique(events[:, 2]).astype(int)
        return {f"event_{code}": int(code) for code in event_codes}

    def _extract_epoch_metadata(self):
        """Extract metadata for each epoch from FIF file."""
        self.epoch_metadata = {}

        # Get events and annotations
        events = self.events
        annotations = self.annotations

        for epoch_idx in range(len(events)):
            event = events[epoch_idx]

            # Find matching annotation (if any)
            epoch_annotation = None
            if annotations is not None:
                # Find annotation closest to this epoch's onset
                epoch_onset = event[0] / self.sfreq  # Convert to seconds
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

    def _determine_marker_type(self, marker_name: str) -> str:
        """Determine marker type from name."""
        name_lower = marker_name.lower()
        if "spindles" in name_lower:
            return "sleep_spindles"
        elif "slowwaves" in name_lower or "slow_waves" in name_lower:
            return "sleep_slow_waves"
        elif "wsmi" in name_lower or "smi" in name_lower:
            return "connectivity"
        elif "psd" in name_lower or "spectral" in name_lower:
            return "spectral"
        elif "pe" in name_lower or "permutation" in name_lower:
            return "information_theory"
        elif "kolmogorov" in name_lower:
            return "information_theory"
        elif any(x in name_lower for x in ["p1", "n1", "p2", "p3"]):
            return "time_locked"
        else:
            return "unknown"

    def _get_actual_channel_names(
        self, marker_data: np.ndarray, marker_type: str
    ) -> List[str]:
        """Infer channel names directly from the tensor dimensions returned by Junifer."""
        if marker_data.size == 0:
            return self.channel_names

        n_epochs = self.n_epochs
        eeg_channels = [
            ch
            for ch in self.channel_names
            if not ch.startswith("EOG") and ch not in ["VEOG", "HEOG"]
        ]

        if marker_data.ndim >= 2 and marker_data.shape[0] == n_epochs:
            # Special handling for sleep markers with 3D structure (epochs, features, channels)
            if marker_type in ["sleep_spindles", "sleep_slow_waves"] and marker_data.ndim == 3:
                n_channels_in_data = marker_data.shape[2]  # channels are in dim 2 for sleep markers
            else:
                n_channels_in_data = marker_data.shape[1]  # default: channels in dim 1

            if marker_type in ["connectivity", "wsmi", "smi"]:
                return eeg_channels[:n_channels_in_data]

            if n_channels_in_data <= len(self.channel_names):
                return self.channel_names[:n_channels_in_data]
            return self.channel_names

        # Fallback to EEG-only ordering
        return eeg_channels

    def _reshape_marker_data(
        self, marker_name: str, data: np.ndarray, marker_type: str
    ) -> np.ndarray:
        """Respect the tensor layout provided by Junifer (epochs already ordered)."""
        logger.info(
            f"Validating {marker_name}: input shape {data.shape}, type {marker_type}"
        )

        if data.size == 0:
            return data

        n_epochs = self.n_epochs

        # Handle sleep markers which have shape (n_features, n_epochs, n_channels)
        if marker_type in ["sleep_spindles", "sleep_slow_waves"]:
            # Handle legacy flattened vector format from old junifer storage layer
            if data.ndim == 2 and data.shape[1] == 1:
                # Check if this is legacy flattened sleep marker data (features x epochs, 1)
                total_elements = data.shape[0]
                if total_elements % n_epochs == 0:
                    n_features = total_elements // n_epochs
                    logger.info(f"Reshaping legacy flattened sleep marker from {data.shape} to ({n_features}, {n_epochs}, 1)")
                    # Reshape from (features x epochs, 1) to (features, epochs, 1)
                    data = data.reshape(n_features, n_epochs, 1)
                    logger.info(f"After reshape: {data.shape}")
                else:
                    logger.warning(
                        f"{marker_name} legacy flattened data size {total_elements} is not divisible by epochs {n_epochs}"
                    )

            # Handle 2D tensor format from updated junifer markers (features, epochs)
            elif data.ndim == 2 and data.shape[1] == n_epochs:
                # This is the new format: (features, epochs) - add missing channel dimension
                n_features = data.shape[0]
                logger.info(f"Adding channel dimension to 2D sleep marker from {data.shape} to ({n_features}, {n_epochs}, 1)")
                data = data[:, :, np.newaxis]  # (features, epochs, 1)
                logger.info(f"After adding channel dimension: {data.shape}")
                # Immediately transpose to (epochs, features, channels) for proper processing
                logger.info(f"Transposing 2D sleep marker data from {data.shape} to (epochs, features, channels)")
                data = np.transpose(data, (1, 0, 2))  # (epochs, features, channels)
                logger.info(f"After transpose: {data.shape}")

            # Handle proper 3D tensor format from updated junifer markers
            elif data.ndim == 3 and (data.shape[1] == n_epochs or data.shape[1] == 1):
                # Transpose from (features, epochs, channels) to (epochs, features, channels)
                # Handle both regular epochs (shape[1] == n_epochs) and aggregated data (shape[1] == 1)
                logger.info(f"Transposing sleep marker data from {data.shape} to (epochs, features, channels)")
                data = np.transpose(data, (1, 0, 2))  # (epochs, features, channels)
                logger.info(f"After transpose: {data.shape}")
            elif data.shape[0] == n_epochs:
                # Already in correct format (epochs, ...)
                pass
            else:
                logger.warning(
                    f"{marker_name} sleep marker has unexpected shape {data.shape}, expected ({n_epochs}, ...) or (features, {n_epochs}, channels) or (features, 1, channels)"
                )

        # Handle spectral power markers which have shape (1, n_bands, n_epochs, n_channels)
        elif "spectral" in marker_type.lower() or "psd" in marker_type.lower():
            logger.info(f"Processing spectral power marker: {marker_name}, shape: {data.shape}")

            # Handle 4D format from junifer storage: (elements, bands, epochs, channels)
            if data.ndim == 4 and data.shape[0] == 1:
                # Squeeze the elements dimension: (1, bands, epochs, channels) → (bands, epochs, channels)
                logger.info(f"Squeezing spectral power data from {data.shape} to {data.shape[1:]}")
                data = data.squeeze(0)  # (bands, epochs, channels)
                logger.info(f"After squeeze: {data.shape}")

                # Transpose from (bands, epochs, channels) to (epochs, bands, channels)
                logger.info(f"Transposing spectral power data from {data.shape} to (epochs, bands, channels)")
                data = np.transpose(data, (1, 0, 2))  # (epochs, bands, channels)
                logger.info(f"After transpose: {data.shape}")

            # Handle 3D format if already squeezed: (bands, epochs, channels)
            elif data.ndim == 3 and data.shape[1] == n_epochs:
                # Transpose from (bands, epochs, channels) to (epochs, bands, channels)
                logger.info(f"Transposing 3D spectral power data from {data.shape} to (epochs, bands, channels)")
                data = np.transpose(data, (1, 0, 2))  # (epochs, bands, channels)
                logger.info(f"After transpose: {data.shape}")

            else:
                logger.warning(
                    f"{marker_name} spectral power marker has unexpected shape {data.shape}, expected (1, bands, {n_epochs}, channels) or (bands, {n_epochs}, channels)"
                )

        elif data.shape[0] != n_epochs:
            logger.warning(
                f"{marker_name} first dimension ({data.shape[0]}) does not match number of epochs ({n_epochs}). "
                "Data will be broadcast across epochs."
            )
            if data.ndim == 1:
                return np.tile(data, (n_epochs, 1))
            return np.broadcast_to(data, (n_epochs, *data.shape))

        return data

    def _normalize_feature_array(
        self, feature_payload: Dict[str, Any]
    ) -> np.ndarray:
        """Extract the ndarray from the HDF5 payload (Junifer now wraps tensors in lists)."""
        data = feature_payload.get("data", [])

        if isinstance(data, list):
            if not data:
                return np.array([])
            if len(data) > 1:
                logger.warning(
                    "Feature contains %d tensors, using the first entry.",
                    len(data),
                )
            data = data[0]

        return np.asarray(data)

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
                "n_epochs": self.n_epochs,
                "n_channels": len(self.channel_names),
                "channel_names": self.channel_names,
                "sfreq": self.sfreq,
                "tmin": self.tmin,
                "tmax": self.tmax,
                "event_id": self.event_id,
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

            try:
                # Handle duplicate names by using MD5 feature ID
                if len(feature_list) > 1:
                    logger.warning(
                        f"Multiple instances of {marker_name} found, using first one with MD5: {feature_list[0][0]}"
                    )
                    feature_id, feature_meta = feature_list[0]
                    # Read using MD5 to avoid duplicate name error
                    feature_data = self.storage.read(feature_md5=feature_id)
                else:
                    feature_id, feature_meta = feature_list[0]
                    # Try reading by name first, fallback to MD5 if needed
                    try:
                        feature_data = self.storage.read(
                            feature_name=marker_name
                        )
                    except Exception:
                        logger.warning(
                            f"Reading by name failed for {marker_name}, trying MD5: {feature_id}"
                        )
                        feature_data = self.storage.read(
                            feature_md5=feature_id
                        )

                data = self._normalize_feature_array(feature_data)

                # Log data info for debugging
                logger.info(f"  Data shape: {data.shape}, dtype: {data.dtype}")
                if data.size > 0:
                    logger.info(
                        f"  Data range: {np.min(data):.6f} to {np.max(data):.6f}"
                    )
                else:
                    logger.warning(f"  *** EMPTY DATA for {marker_name} ***")

                # Determine marker type and reshape data
                marker_type = self._determine_marker_type(marker_name)
                reshaped_data = self._reshape_marker_data(
                    marker_name, data, marker_type
                )

                # Determine actual channel names from reshaped data dimensions
                actual_channel_names = self._get_actual_channel_names(
                    reshaped_data, marker_type
                )

                # Create marker data container
                marker_data = MarkerData(
                    marker_name=marker_name,
                    marker_type=marker_type,
                    channel_names=actual_channel_names,
                    n_epochs=self.n_epochs,
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
                if (
                    reshaped_data.ndim >= 2
                    and reshaped_data.shape[0] == self.n_epochs
                ):
                    # Data is structured as (epochs, ...)
                    for epoch_idx in range(self.n_epochs):
                        epoch_metadata = self.epoch_metadata[epoch_idx]

                        if reshaped_data.ndim == 2:
                            epoch_data_slice = reshaped_data[epoch_idx, :]
                        elif reshaped_data.ndim == 3:
                            epoch_data_slice = reshaped_data[epoch_idx, :, :]  # (features, channels)
                            logger.info(f"DEBUG: Main processing - epoch {epoch_idx}, slice shape: {epoch_data_slice.shape}, marker_type: {marker_type}")
                        else:
                            epoch_data_slice = reshaped_data[epoch_idx]

                        marker_data.add_epoch_data(
                            epoch_idx, epoch_metadata, epoch_data_slice
                        )
                elif (
                    reshaped_data.ndim == 1
                    and len(reshaped_data) % self.n_epochs == 0
                ):
                    # Single flattened array - divide by epochs
                    elements_per_epoch = len(reshaped_data) // self.n_epochs
                    for epoch_idx in range(self.n_epochs):
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
                    # Special handling for sleep markers with aggregated data
                    logger.info(f"DEBUG: Entering special handling - reshaped_data.shape: {reshaped_data.shape}, ndim: {reshaped_data.ndim}, marker_type: {marker_type}")
                    if marker_type in ["sleep_spindles", "sleep_slow_waves"] and reshaped_data.ndim == 3:
                        logger.info(f"DEBUG: Special handling condition TRUE - shape: {reshaped_data.shape}")
                        # Handle both epoch-aggregated (1, n_features, n_channels) and channel-aggregated (n_epochs, n_features, 1)
                        if reshaped_data.shape[0] == 1:
                            # Epoch-aggregated data: (1, n_features, n_channels)
                            feature_data = reshaped_data[0, :, :]  # (n_features, n_channels)
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                # Create epoch data with all channels at once
                                epoch_data = EpochData(epoch_idx, epoch_metadata, actual_channel_names)
                                # Add each channel's feature vector
                                for ch_idx, ch_name in enumerate(actual_channel_names):
                                    if ch_idx < feature_data.shape[1]:
                                        channel_features = feature_data[:, ch_idx]  # (n_features,)
                                        epoch_data.add_channel_data(
                                            ch_name, ch_idx, channel_features
                                        )
                                    else:
                                        # Handle missing channels
                                        epoch_data.add_channel_data(
                                            ch_name, ch_idx, np.zeros(feature_data.shape[0])
                                        )
                                # Store the completed epoch data
                                marker_data._epoch_data[epoch_idx] = epoch_data
                        elif reshaped_data.shape[2] == 1:
                            # Channel-aggregated data: (n_epochs, n_features, 1)
                            logger.info(f"DEBUG: Processing channel-aggregated sleep marker data - shape: {reshaped_data.shape}")
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                # Create epoch data with single channel
                                epoch_data = EpochData(epoch_idx, epoch_metadata, actual_channel_names)
                                # Extract features for this epoch: (n_features,)
                                channel_features = reshaped_data[epoch_idx, :, 0]  # (n_features,)
                                # Add the single channel with all features
                                if len(actual_channel_names) > 0:
                                    epoch_data.add_channel_data(
                                        actual_channel_names[0], 0, channel_features
                                    )
                                # Store the completed epoch data
                                marker_data._epoch_data[epoch_idx] = epoch_data
                        else:
                            # Fallback to original logic
                            logger.info(f"DEBUG: Fallback - shape doesn't match aggregated patterns: {reshaped_data.shape}")
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                marker_data.add_epoch_data(
                                    epoch_idx, epoch_metadata, reshaped_data
                                )
                    else:
                        logger.info(f"DEBUG: Special handling condition FALSE - shape: {reshaped_data.shape}, ndim: {reshaped_data.ndim}, marker_type: {marker_type}")
                        # Original replication logic for non-sleep markers
                        for epoch_idx in range(self.n_epochs):
                            epoch_metadata = self.epoch_metadata[epoch_idx]
                            marker_data.add_epoch_data(
                                epoch_idx, epoch_metadata, reshaped_data
                            )

                # Add feature names for sleep markers (backward compatible) - AFTER data processing
                if marker_type == "sleep_spindles":
                    # Detect actual feature count from the first epoch's channel data
                    if marker_data.keys() and len(marker_data.keys()) > 0:
                        first_epoch_idx = next(iter(marker_data.keys()))
                        first_epoch = marker_data[first_epoch_idx]
                        if first_epoch.keys() and len(first_epoch.keys()) > 0:
                            first_channel = next(iter(first_epoch.keys()))
                            channel_data = first_epoch[first_channel]

                            if channel_data.data.ndim >= 1:
                                n_features = channel_data.data.shape[0]
                                if n_features == 3:
                                    feature_names = ["Duration", "Amplitude", "Frequency"]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected spindles (seconds)",
                                        "Amplitude": "Mean amplitude of detected spindles (µV)",
                                        "Frequency": "Mean frequency of detected spindles (Hz)"
                                    }
                                elif n_features == 4:
                                    feature_names = ["Duration", "Amplitude", "Frequency", "Density"]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected spindles (seconds)",
                                        "Amplitude": "Mean amplitude of detected spindles (µV)",
                                        "Frequency": "Mean frequency of detected spindles (Hz)",
                                        "Density": "Count of detected spindles per epoch/channel"
                                    }
                                else:
                                    feature_names = [f"Feature_{i}" for i in range(n_features)]
                                    feature_descriptions = {f"Feature_{i}": f"Spindle feature {i}" for i in range(n_features)}
                                    logger.warning(f"Unexpected spindles feature count: {n_features}")
                                marker_data.metadata["feature_names"] = feature_names
                                marker_data.metadata["feature_descriptions"] = feature_descriptions
                                marker_data.metadata["n_features"] = n_features

                elif marker_type == "sleep_slow_waves":
                    # Detect actual feature count from the first epoch's channel data
                    if marker_data.keys() and len(marker_data.keys()) > 0:
                        first_epoch_idx = next(iter(marker_data.keys()))
                        first_epoch = marker_data[first_epoch_idx]
                        if first_epoch.keys() and len(first_epoch.keys()) > 0:
                            first_channel = next(iter(first_epoch.keys()))
                            channel_data = first_epoch[first_channel]

                            if channel_data.data.ndim >= 1:
                                n_features = channel_data.data.shape[0]
                                if n_features == 4:
                                    feature_names = ["Duration", "PTP", "Frequency", "Slope"]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected slow waves (seconds)",
                                        "PTP": "Mean peak-to-peak amplitude of detected slow waves (µV)",
                                        "Frequency": "Mean frequency of detected slow waves (Hz)",
                                        "Slope": "Mean slope of detected slow waves"
                                    }
                                elif n_features == 5:
                                    feature_names = ["Duration", "PTP", "Frequency", "Slope", "Density"]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected slow waves (seconds)",
                                        "PTP": "Mean peak-to-peak amplitude of detected slow waves (µV)",
                                        "Frequency": "Mean frequency of detected slow waves (Hz)",
                                        "Slope": "Mean slope of detected slow waves",
                                        "Density": "Count of detected slow waves per epoch/channel"
                                    }
                                else:
                                    feature_names = [f"Feature_{i}" for i in range(n_features)]
                                    feature_descriptions = {f"Feature_{i}": f"Slow wave feature {i}" for i in range(n_features)}
                                    logger.warning(f"Unexpected slow waves feature count: {n_features}")
                                marker_data.metadata["feature_names"] = feature_names
                                marker_data.metadata["feature_descriptions"] = feature_descriptions
                                marker_data.metadata["n_features"] = n_features

                markers[marker_name] = marker_data

            except Exception as e:
                logger.error(f"Failed to process marker {marker_name}: {e}")
                logger.error(
                    f"  Feature IDs: {[fid for fid, _ in feature_list]}"
                )
                # Create empty marker to maintain structure
                empty_marker = MarkerData(
                    marker_name=marker_name,
                    marker_type="unknown",
                    channel_names=self.channel_names,
                    n_epochs=self.n_epochs,
                )
                empty_marker.metadata["error"] = str(e)
                empty_marker.metadata["raw_data"] = np.array([])
                markers[marker_name] = empty_marker
                continue

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
