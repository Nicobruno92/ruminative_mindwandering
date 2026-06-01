#!/usr/bin/env python3
"""
H5 to PKL Converter with FIF Metadata Integration

This converter reads marker results from Junifer HDF5 files and combines them
with epoch metadata from MNE FIF files to create a comprehensive PKL file
with hierarchical indexing: markers -> epochs -> channels with full metadata.

Usage (Modern):
    python h5_to_pkl_converter.py --h5-file <h5_file> --output-dir <dir> [--force]
    
Usage (Legacy):
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
        elif data.ndim == 2:  # Channel x features (spectral, time-series, or single values)
            if self.marker_type in ["sleep_spindles", "sleep_slow_waves"]:
                # Sleep markers: data is (features, channels) after transpose
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if (
                        ch_idx < data.shape[1]
                    ):  # channels are in dim 1 for sleep markers
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
            elif (
                "spectral" in self.marker_type.lower()
                or "psd" in self.marker_type.lower()
            ):
                # Spectral power markers: data is (bands, channels) after transpose
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if (
                        ch_idx < data.shape[1]
                    ):  # channels are in dim 1 for spectral markers
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
            elif self.marker_type == "time_locked":
                # Time-locked ERP markers: data is (channels, timepoints) after epoch slice
                logger.info(
                    f"DEBUG: Processing time-locked marker - shape: {data.shape} (channels, timepoints)"
                )
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if ch_idx < data.shape[0]:  # channels in dim 0
                        # Extract time series for this channel: data[ch_idx, :]
                        channel_timeseries = data[ch_idx, :]  # (n_timepoints,)
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, channel_timeseries
                        )
                    else:
                        # Handle missing channels
                        epoch_data.add_channel_data(
                            ch_name, ch_idx, np.zeros(data.shape[1])
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
        ):  # 3D data - could be connectivity, sleep markers, or time-locked ERPs
            logger.info(
                f"DEBUG: 3D data in add_epoch_data - shape: {data.shape}, marker_type: {self.marker_type}"
            )
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
                logger.info(
                    f"DEBUG: Processing 3D sleep marker data - shape: {data.shape}"
                )
                # Sleep markers: (epochs, features, channels) after transpose
                # Need to extract data[epoch_idx, :, ch_idx] for each channel
                for ch_idx, ch_name in enumerate(self.channel_names):
                    if (
                        ch_idx < data.shape[2]
                    ):  # channels are in dim 2 for sleep markers
                        # Extract all features for this channel: data[epoch_idx, :, ch_idx]
                        channel_features = data[
                            :, :, ch_idx
                        ]  # (epochs, features)
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
        """Infer channel names directly from the tensor dimensions returned by Junifer.
        
        When channel_aggregation_method is used, the channel dimension is removed.
        In that case, we return a single pseudo-channel name ["aggregated"].
        """
        if marker_data.size == 0:
            return self.channel_names

        eeg_channels = [
            ch
            for ch in self.channel_names
            if not ch.startswith("EOG") and ch not in ["VEOG", "HEOG"]
        ]

        # Handle 1D data (channel-aggregated markers)
        if marker_data.ndim == 1:
            # Data is (epochs,) - channel dimension was aggregated
            return ["aggregated"]

        # Don't assume marker_data.shape[0] must equal self.n_epochs
        # The HDF5 data might have a different number of epochs
        if marker_data.ndim >= 2:
            # Special handling for 3D markers: different layouts per marker type
            if marker_data.ndim == 3:
                if marker_type == "time_locked":
                    # Time-locked markers: (epochs, channels, timepoints)
                    n_channels_in_data = marker_data.shape[1]  # channels in dim 1
                else:
                    # Sleep markers: (epochs, features, channels)
                    # Spectral markers: (epochs, bands, channels)
                    n_channels_in_data = marker_data.shape[2]  # channels in dim 2
            else:
                # 2D markers: (epochs, channels) or (epochs, 1) if channel-aggregated
                n_channels_in_data = marker_data.shape[1]  # channels in dim 1
            
            # If only 1 channel, it's likely aggregated
            if n_channels_in_data == 1:
                return ["aggregated"]

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
        """Interpret the tensor layout from Junifer WITHOUT assuming self.n_epochs matches.

        Key insight: We must detect dimensions from the data itself, not from FIF metadata.
        """
        logger.info(
            f"Validating {marker_name}: input shape {data.shape}, type {marker_type}"
        )

        if data.size == 0:
            return data

        # CRITICAL: Detect marker format from shape, not from self.n_epochs
        # Junifer returns different shapes depending on marker type:
        #
        # WITHOUT channel_aggregation_method:
        #   SpectralPower (multi-band):  (n_bands, n_epochs, n_channels)
        #   PermutationEntropy:          (n_epochs, n_channels)
        #   SymbolicMutualInformation:   (n_epochs, n_channels, n_channels)
        #   KolmogorovComplexity:        (n_epochs, n_channels)
        #   SlowWavesDetection:          (n_features, n_epochs, n_channels)
        #   SpindlesDetection:           (n_features, n_epochs, n_channels)
        #   TimeLockedTopography:        (n_epochs, n_channels, n_timepoints)
        #
        # WITH channel_aggregation_method: "mean" (channel dim removed):
        #   SpectralPower (multi-band):  (n_bands, n_epochs)
        #   PermutationEntropy:          (n_epochs,)
        #   SymbolicMutualInformation:   (n_epochs, n_channels) or (n_epochs,)
        #   KolmogorovComplexity:        (n_epochs,)
        #   SlowWavesDetection:          (n_features, n_epochs)
        #   SpindlesDetection:           (n_features, n_epochs)
        #   TimeLockedTopography:        (n_epochs, n_timepoints)

        # Handle sleep markers: (n_features, n_epochs, n_channels) or (n_features, n_epochs) if channel-aggregated
        if marker_type in ["sleep_spindles", "sleep_slow_waves"]:
            if data.ndim == 3:
                # Expected: (n_features, n_epochs, n_channels)
                # Need to transpose to: (n_epochs, n_features, n_channels)
                n_features, n_epochs_in_data, n_channels = data.shape
                logger.info(
                    f"Sleep marker {marker_name}: detected shape (features={n_features}, epochs={n_epochs_in_data}, channels={n_channels})"
                )
                logger.info(
                    "Transposing from (features, epochs, channels) to (epochs, features, channels)"
                )
                data = np.transpose(
                    data, (1, 0, 2)
                )  # (epochs, features, channels)
                logger.info(f"After transpose: {data.shape}")
            elif data.ndim == 2:
                # Channel-aggregated: (n_features, n_epochs) → add pseudo-channel dim
                n_features, n_epochs_in_data = data.shape
                logger.info(
                    f"Sleep marker {marker_name}: channel-aggregated 2D shape (features={n_features}, epochs={n_epochs_in_data})"
                )
                data = data[:, :, np.newaxis]  # (features, epochs, 1)
                data = np.transpose(data, (1, 0, 2))  # (epochs, features, 1)
                logger.info(
                    f"After adding pseudo-channel dim and transpose: {data.shape}"
                )
            else:
                logger.warning(
                    f"Sleep marker {marker_name} has unexpected shape {data.shape}"
                )

        # Handle spectral power markers
        # With channels: (n_bands, n_epochs, n_channels) → (epochs, bands, channels)
        # Channel-aggregated: (n_bands, n_epochs) → (epochs, bands, 1)
        # Flattened channel-aggregated: (n_bands * n_epochs, 1) → (epochs, bands, 1)
        elif "spectral" in marker_type.lower() or "psd" in marker_type.lower():
            logger.info(
                f"Processing spectral marker: {marker_name}, shape: {data.shape}"
            )

            if data.ndim == 4 and data.shape[0] == 1:
                # Format: (1, n_bands, n_epochs, n_channels) - squeeze and transpose
                data = data.squeeze(0)  # (bands, epochs, channels)
                logger.info(f"Squeezed 4D spectral data to: {data.shape}")
                n_bands, n_epochs_in_data, n_channels = data.shape
                logger.info(
                    f"Detected shape (bands={n_bands}, epochs={n_epochs_in_data}, channels={n_channels})"
                )
                # Transpose to (epochs, bands, channels)
                data = np.transpose(data, (1, 0, 2))
                logger.info(
                    f"Transposed to (epochs, bands, channels): {data.shape}"
                )

            elif data.ndim == 3:
                # Detect format: could be (bands, epochs, channels) or (epochs, bands, channels)
                # Heuristic: First dimension is usually bands (smaller) or epochs (larger)
                dim0, dim1, dim2 = data.shape

                # If dim0 is small (2-10), likely bands → transpose needed
                # If dim1 is small (2-10), likely bands → already correct
                if dim0 < 20 and dim1 > dim0:
                    # Likely (bands, epochs, channels) → need transpose
                    logger.info(
                        f"Detected (bands={dim0}, epochs={dim1}, channels={dim2}) - transposing"
                    )
                    data = np.transpose(
                        data, (1, 0, 2)
                    )  # (epochs, bands, channels)
                    logger.info(f"After transpose: {data.shape}")
                else:
                    # Likely already (epochs, bands, channels) or (epochs, channels, features)
                    logger.info(
                        f"Shape appears to be (epochs={dim0}, dim1={dim1}, dim2={dim2}) - keeping as is"
                    )

            elif data.ndim == 2:
                # Could be:
                # 1. (n_bands, n_epochs) - channel aggregated, need transpose
                # 2. (n_epochs, n_channels) - single band
                # 3. (n_bands * n_epochs, 1) - flattened channel-aggregated (Junifer quirk)
                dim0, dim1 = data.shape
                
                # Check for flattened data: (n_bands * n_epochs, 1)
                if dim1 == 1 and dim0 > self.n_epochs:
                    # Likely flattened (epochs * bands, 1)
                    # Common band counts: 5 (delta, theta, alpha, beta, gamma)
                    n_bands_guess = dim0 // self.n_epochs
                    if dim0 == self.n_epochs * n_bands_guess:
                        logger.info(
                            f"Flattened spectral data detected: ({dim0}, {dim1}) = ({self.n_epochs} epochs * {n_bands_guess} bands, 1)"
                        )
                        # Reshape: (epochs * bands, 1) → (epochs, bands, 1)
                        # Junifer stores as [e0b0, e0b1, ..., e0bN, e1b0, e1b1, ...] or
                        # [e0b0, e1b0, e2b0, ..., e0b1, e1b1, ...] - need to check
                        # Assuming row-major: each epoch's bands are contiguous
                        data = data.reshape(self.n_epochs, n_bands_guess, 1)
                        logger.info(f"Reshaped to (epochs, bands, 1): {data.shape}")
                    else:
                        logger.warning(
                            f"Cannot reshape flattened data: {dim0} is not divisible by n_epochs={self.n_epochs}"
                        )
                elif dim0 < 20 and dim1 > dim0:
                    # Likely (bands, epochs) - channel aggregated
                    logger.info(
                        f"Channel-aggregated spectral data: (bands={dim0}, epochs={dim1}) - adding pseudo-channel"
                    )
                    data = data[:, :, np.newaxis]  # (bands, epochs, 1)
                    data = np.transpose(data, (1, 0, 2))  # (epochs, bands, 1)
                    logger.info(f"After transpose: {data.shape}")
                else:
                    # Single band: (epochs, channels) or (epochs, 1) if aggregated
                    logger.info(
                        f"Single-band spectral data: {data.shape} (epochs, channels or aggregated)"
                    )

            elif data.ndim == 1:
                # Fully aggregated: (n_epochs,) → add pseudo-channel dim
                logger.info(
                    f"Fully aggregated spectral data: {data.shape} - adding pseudo-channel"
                )
                data = data[:, np.newaxis]  # (epochs, 1)

            else:
                logger.warning(
                    f"Spectral marker {marker_name} has unexpected shape {data.shape}"
                )

        # Handle connectivity markers
        # With channels: (n_epochs, n_channels, n_channels)
        # connectivity_aggregation only: (n_epochs, n_channels)
        # channel_aggregation only: (n_epochs, n_channels) - same as above
        # Both aggregations: (n_epochs,)
        elif marker_type in ["connectivity", "wsmi", "smi"]:
            if data.ndim == 3:
                # Full connectivity matrix: (n_epochs, n_channels, n_channels)
                n_epochs_in_data, n_ch1, n_ch2 = data.shape
                logger.info(
                    f"Connectivity marker {marker_name}: shape (epochs={n_epochs_in_data}, ch1={n_ch1}, ch2={n_ch2})"
                )
            elif data.ndim == 2:
                # Aggregated connectivity (connectivity_aggregation_method or channel_aggregation_method): (n_epochs, n_channels)
                n_epochs_in_data, n_channels = data.shape
                logger.info(
                    f"Connectivity marker {marker_name}: aggregated shape (epochs={n_epochs_in_data}, channels={n_channels})"
                )
            elif data.ndim == 1:
                # Fully aggregated (both connectivity and channel aggregation): (n_epochs,)
                n_epochs_in_data = data.shape[0]
                logger.info(
                    f"Connectivity marker {marker_name}: fully aggregated shape (epochs={n_epochs_in_data}) - adding pseudo-channel"
                )
                data = data[:, np.newaxis]  # (epochs, 1)
            else:
                logger.warning(
                    f"Connectivity marker {marker_name} has unexpected shape {data.shape}"
                )

        # Handle time-locked markers - compute mean across timepoints for 2D output
        # With channels: (n_epochs, n_channels, n_timepoints) → (epochs, channels)
        # Channel-aggregated: (n_epochs, n_timepoints) → (epochs, 1)
        elif marker_type == "time_locked":
            if data.ndim == 3:
                n_epochs_in_data, n_channels, n_timepoints = data.shape
                logger.info(
                    f"Time-locked marker {marker_name}: shape (epochs={n_epochs_in_data}, channels={n_channels}, timepoints={n_timepoints})"
                )
                logger.info(
                    "Computing mean across timepoints axis to get 2D data (epochs, channels) - simplified approach"
                )
                # Compute mean across timepoints (axis=2) to get (epochs, channels)
                # This provides mean ERP amplitude per channel, which is scientifically standard
                data = np.mean(data, axis=2)
                logger.info(
                    f"After timepoints averaging: new shape (epochs={data.shape[0]}, channels={data.shape[1]})"
                )
            elif data.ndim == 2:
                # Channel-aggregated: (n_epochs, n_timepoints) → compute mean over timepoints
                n_epochs_in_data, n_timepoints = data.shape
                logger.info(
                    f"Time-locked marker {marker_name}: channel-aggregated shape (epochs={n_epochs_in_data}, timepoints={n_timepoints})"
                )
                # Compute mean across timepoints (axis=1) to get (epochs,)
                data = np.mean(data, axis=1)
                logger.info(
                    f"After timepoints averaging: shape (epochs={data.shape[0]}) - adding pseudo-channel"
                )
                data = data[:, np.newaxis]  # (epochs, 1)
            elif data.ndim == 1:
                # Fully aggregated: (n_epochs,) → add pseudo-channel
                logger.info(
                    f"Time-locked marker {marker_name}: fully aggregated shape {data.shape} - adding pseudo-channel"
                )
                data = data[:, np.newaxis]  # (epochs, 1)
            else:
                raise ValueError(
                    f"Time-locked marker {marker_name} has unexpected shape {data.shape} with {data.ndim} dimensions."
                )

        # Handle simple 2D markers: (epochs, channels)
        elif data.ndim == 2:
            n_epochs_in_data, n_channels = data.shape
            logger.info(
                f"2D marker {marker_name}: shape (epochs={n_epochs_in_data}, channels={n_channels})"
            )

        # Handle simple 1D markers: (epochs,) - channel aggregated
        elif data.ndim == 1:
            n_epochs_in_data = data.shape[0]
            logger.info(
                f"1D marker {marker_name}: channel-aggregated shape (epochs={n_epochs_in_data}) - adding pseudo-channel"
            )
            data = data[:, np.newaxis]  # (epochs, 1)

        # Handle 1D aggregated markers
        elif data.ndim == 1:
            logger.info(
                f"1D marker {marker_name}: shape {data.shape} (aggregated)"
            )

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

                # Detect actual number of epochs in HDF5 data (may differ from FIF)
                actual_n_epochs_in_data = (
                    reshaped_data.shape[0] if reshaped_data.ndim >= 1 else 1
                )

                # Create marker data container
                # Use n_epochs from FIF for metadata structure, but actual data will come from HDF5
                marker_data = MarkerData(
                    marker_name=marker_name,
                    marker_type=marker_type,
                    channel_names=actual_channel_names,
                    n_epochs=actual_n_epochs_in_data,  # Use actual epochs from HDF5 data
                )

                # Store the raw reshaped data directly in metadata for validation
                marker_data.metadata["raw_data"] = reshaped_data

                # Add epoch data - handle different data structures
                if reshaped_data.size == 0:
                    logger.error(
                        f"No data to process for {marker_name} - skipping"
                    )
                    continue

                # Detect the actual number of epochs in the HDF5 data
                # This may differ from self.n_epochs (from FIF) if Junifer processed a subset
                actual_n_epochs_in_data = (
                    reshaped_data.shape[0] if reshaped_data.ndim >= 1 else 1
                )

                logger.info(
                    f"{marker_name}: HDF5 data has {actual_n_epochs_in_data} epochs, "
                    f"FIF has {self.n_epochs} epochs"
                )

                # Check if we have per-epoch data
                if (
                    reshaped_data.ndim >= 2
                    and reshaped_data.shape[0]
                    > 1  # Use actual data shape, not self.n_epochs
                ):
                    # Data is structured as (epochs, ...)
                    # Use the actual number of epochs from the data
                    for data_epoch_idx in range(actual_n_epochs_in_data):
                        # Map to the correct epoch_idx from FIF metadata
                        # If data has fewer epochs than FIF, use sequential mapping
                        if data_epoch_idx < len(self.epoch_metadata):
                            epoch_metadata = self.epoch_metadata[
                                data_epoch_idx
                            ]
                            actual_epoch_idx = data_epoch_idx
                        else:
                            # Fallback: create minimal metadata for extra epochs
                            logger.warning(
                                f"No FIF metadata for data epoch {data_epoch_idx}, creating fallback"
                            )
                            epoch_metadata = EpochMetadata(
                                epoch_idx=data_epoch_idx,
                                event=np.array([0, 0, 0]),  # Dummy event
                                annotation=None,
                            )
                            actual_epoch_idx = data_epoch_idx

                        if reshaped_data.ndim == 2:
                            epoch_data_slice = reshaped_data[data_epoch_idx, :]
                        elif reshaped_data.ndim == 3:
                            epoch_data_slice = reshaped_data[
                                data_epoch_idx, :, :
                            ]  # (features, channels)
                            logger.info(
                                f"DEBUG: Main processing - epoch {data_epoch_idx}, slice shape: {epoch_data_slice.shape}, marker_type: {marker_type}"
                            )
                        else:
                            epoch_data_slice = reshaped_data[data_epoch_idx]

                        marker_data.add_epoch_data(
                            actual_epoch_idx, epoch_metadata, epoch_data_slice
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
                    logger.info(
                        f"DEBUG: Entering special handling - reshaped_data.shape: {reshaped_data.shape}, ndim: {reshaped_data.ndim}, marker_type: {marker_type}"
                    )
                    if (
                        marker_type in ["sleep_spindles", "sleep_slow_waves"]
                        and reshaped_data.ndim == 3
                    ):
                        logger.info(
                            f"DEBUG: Special handling condition TRUE - shape: {reshaped_data.shape}"
                        )
                        # Handle both epoch-aggregated (1, n_features, n_channels) and channel-aggregated (n_epochs, n_features, 1)
                        if reshaped_data.shape[0] == 1:
                            # Epoch-aggregated data: (1, n_features, n_channels)
                            feature_data = reshaped_data[
                                0, :, :
                            ]  # (n_features, n_channels)
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                # Create epoch data with all channels at once
                                epoch_data = EpochData(
                                    epoch_idx,
                                    epoch_metadata,
                                    actual_channel_names,
                                )
                                # Add each channel's feature vector
                                for ch_idx, ch_name in enumerate(
                                    actual_channel_names
                                ):
                                    if ch_idx < feature_data.shape[1]:
                                        channel_features = feature_data[
                                            :, ch_idx
                                        ]  # (n_features,)
                                        epoch_data.add_channel_data(
                                            ch_name, ch_idx, channel_features
                                        )
                                    else:
                                        # Handle missing channels
                                        epoch_data.add_channel_data(
                                            ch_name,
                                            ch_idx,
                                            np.zeros(feature_data.shape[0]),
                                        )
                                # Store the completed epoch data
                                marker_data._epoch_data[epoch_idx] = epoch_data
                        elif reshaped_data.shape[2] == 1:
                            # Channel-aggregated data: (n_epochs, n_features, 1)
                            logger.info(
                                f"DEBUG: Processing channel-aggregated sleep marker data - shape: {reshaped_data.shape}"
                            )
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                # Create epoch data with single channel
                                epoch_data = EpochData(
                                    epoch_idx,
                                    epoch_metadata,
                                    actual_channel_names,
                                )
                                # Extract features for this epoch: (n_features,)
                                channel_features = reshaped_data[
                                    epoch_idx, :, 0
                                ]  # (n_features,)
                                # Add the single channel with all features
                                if len(actual_channel_names) > 0:
                                    epoch_data.add_channel_data(
                                        actual_channel_names[0],
                                        0,
                                        channel_features,
                                    )
                                # Store the completed epoch data
                                marker_data._epoch_data[epoch_idx] = epoch_data
                        else:
                            # Fallback to original logic
                            logger.info(
                                f"DEBUG: Fallback - shape doesn't match aggregated patterns: {reshaped_data.shape}"
                            )
                            for epoch_idx in range(self.n_epochs):
                                epoch_metadata = self.epoch_metadata[epoch_idx]
                                marker_data.add_epoch_data(
                                    epoch_idx, epoch_metadata, reshaped_data
                                )
                    else:
                        logger.info(
                            f"DEBUG: Special handling condition FALSE - shape: {reshaped_data.shape}, ndim: {reshaped_data.ndim}, marker_type: {marker_type}"
                        )
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
                                    feature_names = [
                                        "Duration",
                                        "Amplitude",
                                        "Frequency",
                                    ]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected spindles (seconds)",
                                        "Amplitude": "Mean amplitude of detected spindles (µV)",
                                        "Frequency": "Mean frequency of detected spindles (Hz)",
                                    }
                                elif n_features == 4:
                                    feature_names = [
                                        "Duration",
                                        "Amplitude",
                                        "Frequency",
                                        "Density",
                                    ]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected spindles (seconds)",
                                        "Amplitude": "Mean amplitude of detected spindles (µV)",
                                        "Frequency": "Mean frequency of detected spindles (Hz)",
                                        "Density": "Count of detected spindles per epoch/channel",
                                    }
                                else:
                                    feature_names = [
                                        f"Feature_{i}"
                                        for i in range(n_features)
                                    ]
                                    feature_descriptions = {
                                        f"Feature_{i}": f"Spindle feature {i}"
                                        for i in range(n_features)
                                    }
                                    logger.warning(
                                        f"Unexpected spindles feature count: {n_features}"
                                    )
                                marker_data.metadata["feature_names"] = (
                                    feature_names
                                )
                                marker_data.metadata[
                                    "feature_descriptions"
                                ] = feature_descriptions
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
                                    feature_names = [
                                        "Duration",
                                        "PTP",
                                        "Frequency",
                                        "Slope",
                                    ]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected slow waves (seconds)",
                                        "PTP": "Mean peak-to-peak amplitude of detected slow waves (µV)",
                                        "Frequency": "Mean frequency of detected slow waves (Hz)",
                                        "Slope": "Mean slope of detected slow waves",
                                    }
                                elif n_features == 5:
                                    feature_names = [
                                        "Duration",
                                        "PTP",
                                        "Frequency",
                                        "Slope",
                                        "Density",
                                    ]
                                    feature_descriptions = {
                                        "Duration": "Mean duration of detected slow waves (seconds)",
                                        "PTP": "Mean peak-to-peak amplitude of detected slow waves (µV)",
                                        "Frequency": "Mean frequency of detected slow waves (Hz)",
                                        "Slope": "Mean slope of detected slow waves",
                                        "Density": "Count of detected slow waves per epoch/channel",
                                    }
                                else:
                                    feature_names = [
                                        f"Feature_{i}"
                                        for i in range(n_features)
                                    ]
                                    feature_descriptions = {
                                        f"Feature_{i}": f"Slow wave feature {i}"
                                        for i in range(n_features)
                                    }
                                    logger.warning(
                                        f"Unexpected slow waves feature count: {n_features}"
                                    )
                                marker_data.metadata["feature_names"] = (
                                    feature_names
                                )
                                marker_data.metadata[
                                    "feature_descriptions"
                                ] = feature_descriptions
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
    """Main function for command-line usage.
    
    Supports two interfaces:
    1. Legacy: h5_to_pkl_converter.py fif_file h5_file output_pkl
    2. Modern: h5_to_pkl_converter.py --h5-file FILE --output-dir DIR [--force]
    """
    parser = argparse.ArgumentParser(
        description="Convert Junifer HDF5 + MNE FIF to organized PKL"
    )
    
    # Modern interface (named arguments)
    parser.add_argument(
        "--h5-file", 
        help="Path to Junifer HDF5 features file"
    )
    parser.add_argument(
        "--fif-file",
        help="Path to MNE epochs FIF file (optional, auto-detected from H5 metadata)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for output PKL files (filename auto-generated)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    # Legacy interface (positional arguments)
    parser.add_argument(
        "fif_file_pos", 
        nargs="?",
        help="[Legacy] Path to MNE epochs FIF file"
    )
    parser.add_argument(
        "h5_file_pos", 
        nargs="?",
        help="[Legacy] Path to Junifer HDF5 features file"
    )
    parser.add_argument(
        "output_pkl_pos", 
        nargs="?",
        help="[Legacy] Path for output PKL file"
    )
    
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

    # Determine which interface is being used
    if args.h5_file:
        # Modern interface
        h5_file = args.h5_file
        fif_file = args.fif_file  # May be None - will auto-detect
        
        if args.output_dir:
            # Auto-generate output filename from H5 filename
            h5_path = Path(h5_file)
            # e.g., element_sub-XX_ses-X_task-X_desc_markers.h5 -> same name with .pkl
            output_pkl = Path(args.output_dir) / h5_path.name.replace(".h5", ".pkl")
        else:
            # Default to same directory as H5
            h5_path = Path(h5_file)
            output_pkl = h5_path.with_suffix(".pkl")
        
        # Check if output exists and force flag
        if output_pkl.exists() and not args.force:
            print(f"Output file already exists: {output_pkl}")
            print("Use --force to overwrite")
            return 0
            
        # Auto-detect FIF file from H5 filename if not provided
        if fif_file is None:
            fif_file = _infer_fif_from_h5(h5_file)
            if fif_file is None:
                logger.error(
                    "Could not auto-detect FIF file from H5 filename. "
                    "Please provide --fif-file explicitly."
                )
                return 1
            logger.info(f"Auto-detected FIF file: {fif_file}")
    
    elif args.fif_file_pos and args.h5_file_pos and args.output_pkl_pos:
        # Legacy interface
        fif_file = args.fif_file_pos
        h5_file = args.h5_file_pos
        output_pkl = args.output_pkl_pos
    
    else:
        parser.print_help()
        print("\nError: Please provide either:")
        print("  1. --h5-file FILE --output-dir DIR")
        print("  2. fif_file h5_file output_pkl (positional)")
        return 1

    try:
        # Create output directory if needed
        output_path = Path(output_pkl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create converter and run conversion
        converter = H5ToPklConverter(fif_file, h5_file)
        data = converter.convert()
        converter.save_pkl(output_pkl, data)

        if args.demo:
            demonstrate_usage(data)

        print("\nConversion successful!")
        print(f"Input FIF: {fif_file}")
        print(f"Input HDF5: {h5_file}")
        print(f"Output PKL: {output_pkl}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def _infer_fif_from_h5(h5_path: str) -> Optional[str]:
    """Infer the corresponding FIF file path from the H5 filename.
    
    H5 pattern: element_sub-XX_ses-X_TASK_DESC_markers.h5
    FIF pattern: sub-XX/ses-X/eeg/sub-XX_ses-X_task-TASK_desc-DESC_epo.fif
    
    Examples:
        element_sub-02_ses-a_sartauditiva_evoked_markers.h5
        -> sub-02/ses-a/eeg/sub-02_ses-a_task-sartauditiva_desc-evoked_epo.fif
    """
    import re
    
    h5_name = Path(h5_path).name
    
    # Parse element_sub-XX_ses-X_TASK_DESC_markers.h5
    # Example: element_sub-02_ses-a_sartauditiva_evoked_markers.h5
    match = re.match(
        r"element_(sub-\d+)_(ses-[ab])_(\w+)_(\w+)_markers\.h5",
        h5_name
    )
    
    if not match:
        # Try alternate pattern with task- prefix
        match = re.match(
            r"element_(sub-\d+)_(ses-[ab])_task-(\w+)_(\w+)_markers\.h5",
            h5_name
        )
        if not match:
            return None
    
    subject, session, task, desc = match.groups()
    
    # Build FIF path
    # Assume derivatives directory is at a known location
    workdir = Path("/network/iss/levy/analyze/valerocabre/analyse/nbruno/wandering-mind")
    derivatives_dir = workdir / "data" / "derivatives"
    
    # Primary pattern: sub-XX_ses-X_task-TASK_desc-DESC_epo.fif
    fif_path = (
        derivatives_dir / subject / session / "eeg" / 
        f"{subject}_{session}_task-{task}_desc-{desc}_epo.fif"
    )
    
    if fif_path.exists():
        return str(fif_path)
    
    # Try alternate naming (state vs evoked, sleep, etc.)
    # The desc might be "evoked", "state", "sleep" 
    for alt_desc in ["evoked", "state", "sleep"]:
        alt_fif_path = (
            derivatives_dir / subject / session / "eeg" / 
            f"{subject}_{session}_task-{task}_desc-{alt_desc}_epo.fif"
        )
        if alt_fif_path.exists():
            return str(alt_fif_path)
    
    return None


if __name__ == "__main__":
    exit(main())
