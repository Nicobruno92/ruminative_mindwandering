#!/usr/bin/env python3
"""
JuniferHDF5Reader - Simple reader for Junifer HDF5 output files (non-aggregated markers).

Usage:
    from junifer_h5_reader import JuniferHDF5Reader

    reader = JuniferHDF5Reader("path/to/output.h5")
    reader.list_markers()

    # Get data
    data = reader.get_marker("PE_theta")

    # Load events and create dataframe with rows=epochs, columns=channels+events
    reader.load_events("events.tsv")
    df = reader.to_dataframe("PE_theta", include_events=True)

    # For multi-band markers, specify band
    df = reader.to_dataframe("psd_bands", band="theta", include_events=True)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from junifer.storage import HDF5FeatureStorage

MARKER_TYPES = {
    "spectral": {"keywords": ["psd", "spectral", "spectralpower"]},
    "entropy": {"keywords": ["pe_", "permutation", "permutationentropy"]},
    "complexity": {"keywords": ["kolmogorov", "kolmogorovcomplexity"]},
    "connectivity": {"keywords": ["wsmi", "smi", "symbolicmutualinformation"]},
    "sleep": {
        "keywords": ["slowwaves", "spindles", "slow_waves"],
        "feature_names": {
            "slowwaves": ["Duration", "PTP", "Frequency", "Slope", "Density"],
            "spindles": ["Duration", "Amplitude", "Frequency", "Density"],
        },
    },
    "erp": {"keywords": ["p1", "n1", "p2", "p3", "timelockedtopo"]},
}

# Separators used in channel pair column names for non-aggregated connectivity
# Support both '--' and '-' formats
CONNECTIVITY_PAIR_SEPARATORS = ["--", "-"]


class JuniferHDF5Reader:
    """Simple reader for Junifer HDF5 output files (non-aggregated markers only)."""

    def __init__(self, h5_path: Union[str, Path]):
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

        self.storage = HDF5FeatureStorage(str(self.path))
        self._features = self.storage.list_features()
        self._cache: Dict[str, Dict] = {}
        self._events_df = None
        self._build_marker_index()

    def _build_marker_index(self):
        """Build index mapping short names to full feature names."""
        self.markers = {}
        self._md5_map = {}
        for md5, meta in self._features.items():
            full_name = meta.get("name", md5)
            short_name = self._extract_short_name(full_name)
            self.markers[short_name] = full_name
            self._md5_map[short_name] = md5

    def _extract_short_name(self, full_name: str) -> str:
        """Extract user-friendly short name from full feature name."""
        name = full_name
        for prefix in ["EEG_", "eeg_"]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        suffixes = [
            "_spectralpower",
            "_permutationentropy",
            "_symbolicmutualinformation",
            "_kolmogorovcomplexity",
            "_timelockedtopo",
            "_slowwavesdetection",
            "_spindlesdetection",
            "_contingentvariation",
        ]
        for suffix in suffixes:
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    def _detect_marker_type(self, name: str) -> str:
        """Detect marker type from name."""
        name_lower = name.lower()
        for mtype, info in MARKER_TYPES.items():
            for keyword in info["keywords"]:
                if keyword in name_lower:
                    return mtype
        return "unknown"

    def list_markers(self) -> List[str]:
        """List all available markers."""
        return list(self.markers.keys())

    def _load_marker(self, marker_name: str) -> Dict:
        """Load marker data and cache it."""
        if marker_name in self._cache:
            return self._cache[marker_name]

        if marker_name not in self.markers:
            for short in self.markers:
                if short.lower() == marker_name.lower():
                    marker_name = short
                    break
            else:
                raise KeyError(
                    f"Marker '{marker_name}' not found. Available: {self.list_markers()}"
                )

        full_name = self.markers[marker_name]
        data_dict = self.storage.read(feature_name=full_name)
        data = data_dict.get("data", np.array([]))

        if isinstance(data, list) and len(data) > 0:
            data = (
                np.asarray(data[0])
                if len(data) == 1
                else np.stack([np.asarray(d) for d in data], axis=0)
            )
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)

        col_names = (
            list(data_dict.get("column_headers", []))
            if data_dict.get("column_headers") is not None
            else []
        )

        # Handle missing column headers - generate generic names based on data shape
        if len(col_names) == 0 and data.ndim >= 2:
            # For connectivity (epochs, channels) or ERP (epochs, channels, times)
            n_channels = data.shape[1]
            col_names = [f"ch_{i}" for i in range(n_channels)]

        marker_type = self._detect_marker_type(full_name)

        # Detect if this is non-aggregated connectivity (has channel pairs)
        is_connectivity_pairs = self._is_connectivity_pairs(col_names)
        channel_names_unique = None
        if is_connectivity_pairs:
            channel_names_unique = self._extract_unique_channels_from_pairs(
                col_names
            )

        info = {
            "name": marker_name,
            "marker_type": marker_type,
            "data": data,
            "col_names": col_names,
            "band_names": self._get_band_names(data, marker_type),
            "feature_names": self._get_feature_names(
                marker_name, marker_type, data
            ),
            "is_connectivity_pairs": is_connectivity_pairs,
            "channel_names_unique": channel_names_unique,
        }
        self._cache[marker_name] = info
        return info

    def _get_band_names(
        self, data: np.ndarray, marker_type: str
    ) -> Optional[List[str]]:
        """Get band names for spectral markers."""
        if marker_type != "spectral" or data.ndim != 3:
            return None
        n_bands = data.shape[0]
        if n_bands == 5:
            return ["delta", "theta", "alpha", "beta", "gamma"]
        if n_bands == 6:
            return ["delta", "theta", "alpha", "beta", "jota", "gamma"]
        return [f"band_{i}" for i in range(n_bands)]

    def _get_feature_names(
        self, name: str, marker_type: str, data: np.ndarray
    ) -> Optional[List[str]]:
        """Get feature names for sleep markers."""
        if marker_type != "sleep" or data.ndim != 3:
            return None
        name_lower = name.lower()
        if "slowwaves" in name_lower or "slow_waves" in name_lower:
            return ["Duration", "PTP", "Frequency", "Slope", "Density"]
        if "spindles" in name_lower:
            return ["Duration", "Amplitude", "Frequency", "Density"]
        return [f"feature_{i}" for i in range(data.shape[0])]

    def _get_pair_separator(self, col_name: str) -> Optional[str]:
        """Get the separator used in a column name, if any."""
        for sep in CONNECTIVITY_PAIR_SEPARATORS:
            if sep in col_name:
                # For single '-', make sure it's not just a channel name with '-'
                # by checking if split gives exactly 2 non-empty parts
                parts = col_name.split(sep)
                if len(parts) == 2 and all(p.strip() for p in parts):
                    return sep
        return None

    def _is_connectivity_pairs(self, col_names: List[str]) -> bool:
        """Check if column names represent connectivity pairs (e.g., 'ch1--ch2' or 'ch1-ch2')."""
        if not col_names or len(col_names) == 0:
            return False
        # Check if at least some columns contain a pair separator
        pair_count = sum(
            1 for c in col_names if self._get_pair_separator(c) is not None
        )
        # Consider it pairs if majority have a separator
        return pair_count > len(col_names) * 0.5

    def _extract_unique_channels_from_pairs(
        self, col_names: List[str]
    ) -> List[str]:
        """Extract unique channel names from connectivity pair column names."""
        channels = set()
        for col in col_names:
            sep = self._get_pair_separator(col)
            if sep:
                parts = col.split(sep)
                if len(parts) == 2:
                    channels.add(parts[0].strip())
                    channels.add(parts[1].strip())
        # Sort to maintain consistent order
        return sorted(channels)

    def _parse_channel_pair(self, col_name: str) -> Optional[tuple]:
        """Parse a channel pair column name into (ch1, ch2) tuple."""
        sep = self._get_pair_separator(col_name)
        if not sep:
            return None
        parts = col_name.split(sep)
        if len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())
        return None

    def get_marker_info(self, marker_name: str) -> Dict:
        """Get full information about a marker (for compatibility)."""
        info = self._load_marker(marker_name)

        class MarkerInfo:
            def __init__(self, d):
                self.name = d["name"]
                self.marker_type = d["marker_type"]
                self.data = d["data"]
                self.col_names = d["col_names"]
                self.band_names = d["band_names"]
                self.feature_names = d["feature_names"]
                self.shape = d["data"].shape
                self.channel_names = d["col_names"]
                self.n_epochs = (
                    d["data"].shape[0] if d["data"].ndim >= 1 else None
                )
                # Non-aggregated connectivity info
                self.is_connectivity_pairs = d.get(
                    "is_connectivity_pairs", False
                )
                self.channel_names_unique = d.get("channel_names_unique", None)
                self.n_channel_pairs = (
                    d["data"].shape[1]
                    if self.is_connectivity_pairs and d["data"].ndim == 2
                    else None
                )
                # Handle different data layouts:
                # - ERP 3D: (epochs, channels, times) -> channels at index 1
                # - Spectral 3D: (bands, epochs, channels) -> channels at index 2
                # - Sleep 3D: (features, epochs, channels) -> channels at index 2
                # - Connectivity pairs 2D: (epochs, channel_pairs) -> n_channels from unique channels
                # - 2D: (epochs, channels) -> channels at index 1
                if self.is_connectivity_pairs:
                    # Non-aggregated connectivity: n_channels from unique channel names
                    self.n_channels = (
                        len(self.channel_names_unique)
                        if self.channel_names_unique
                        else None
                    )
                elif d["data"].ndim == 3:
                    if d["marker_type"] == "erp":
                        # ERP: (epochs, channels, times)
                        self.n_channels = d["data"].shape[1]
                    else:
                        # Spectral/Sleep: (bands/features, epochs, channels)
                        self.n_channels = d["data"].shape[2]
                elif d["data"].ndim >= 2:
                    self.n_channels = d["data"].shape[-1]
                else:
                    self.n_channels = None

        return MarkerInfo(info)

    def get_marker(
        self,
        marker_name: str,
        epoch: Optional[Union[int, List[int], slice]] = None,
        channel: Optional[Union[str, List[str], int]] = None,
        band: Optional[Union[str, int]] = None,
        feature: Optional[Union[str, int]] = None,
        channel_pair: Optional[Union[str, tuple]] = None,
    ) -> np.ndarray:
        """Get marker data with optional slicing."""
        info = self._load_marker(marker_name)
        data = info["data"]
        col_names = info["col_names"]
        marker_type = info["marker_type"]
        is_connectivity_pairs = info.get("is_connectivity_pairs", False)

        band_idx = self._resolve_index(band, info["band_names"], "band")
        feature_idx = self._resolve_index(
            feature, info["feature_names"], "feature"
        )
        epoch_idx = epoch if epoch is not None else slice(None)

        # Handle non-aggregated connectivity (channel pairs)
        if is_connectivity_pairs and marker_type == "connectivity":
            if channel_pair is not None:
                pair_idx = self._resolve_channel_pair_index(
                    channel_pair, col_names
                )
                return data[epoch_idx, pair_idx]
            elif channel is not None:
                # Filter pairs involving the specified channel(s)
                pair_indices = self._get_pairs_for_channel(channel, col_names)
                return (
                    data[epoch_idx][:, pair_indices]
                    if isinstance(epoch_idx, slice)
                    else data[epoch_idx, pair_indices]
                )
            else:
                return data[epoch_idx]

        channel_idx = self._resolve_channel_index(channel, col_names)

        if marker_type == "spectral" and data.ndim == 3:
            return data[band_idx, epoch_idx, channel_idx]
        elif marker_type == "sleep" and data.ndim == 3:
            return data[feature_idx, epoch_idx, channel_idx]
        elif marker_type == "connectivity" and data.ndim == 2:
            return data[epoch_idx, channel_idx]
        elif data.ndim == 2:
            return data[epoch_idx, channel_idx]
        elif data.ndim == 1:
            return data[epoch_idx] if epoch is not None else data
        return data

    def _resolve_index(self, value, names: Optional[List[str]], label: str):
        """Resolve a named or integer index."""
        if value is None:
            return slice(None)
        if isinstance(value, int):
            return value
        if names is None:
            raise ValueError(
                f"{label} names not available. Use integer index."
            )
        value_lower = value.lower()
        for idx, name in enumerate(names):
            if name.lower() == value_lower:
                return idx
        raise ValueError(f"{label} '{value}' not found. Available: {names}")

    def _resolve_channel_index(self, channel, col_names: List[str]):
        """Resolve channel specification to index."""
        if channel is None:
            return slice(None)
        if isinstance(channel, int):
            return channel
        if isinstance(channel, str):
            if channel not in col_names:
                raise ValueError(f"Channel '{channel}' not found.")
            return col_names.index(channel)
        if isinstance(channel, list):
            return np.array(
                [
                    col_names.index(ch) if isinstance(ch, str) else ch
                    for ch in channel
                ]
            )
        return slice(None)

    def _resolve_channel_pair_index(
        self, channel_pair: Union[str, tuple], col_names: List[str]
    ) -> int:
        """Resolve channel pair to column index."""
        if isinstance(channel_pair, tuple):
            ch1, ch2 = channel_pair
            # Try all separator formats
            for sep in CONNECTIVITY_PAIR_SEPARATORS:
                pair_str = f"{ch1}{sep}{ch2}"
                pair_str_rev = f"{ch2}{sep}{ch1}"
                if pair_str in col_names:
                    return col_names.index(pair_str)
                if pair_str_rev in col_names:
                    return col_names.index(pair_str_rev)
        else:
            # Direct string match first
            if channel_pair in col_names:
                return col_names.index(channel_pair)
            # Try to parse and find reverse
            for sep in CONNECTIVITY_PAIR_SEPARATORS:
                if sep in channel_pair:
                    parts = channel_pair.split(sep)
                    if len(parts) == 2:
                        pair_str_rev = f"{parts[1]}{sep}{parts[0]}"
                        if pair_str_rev in col_names:
                            return col_names.index(pair_str_rev)
                    break

        raise ValueError(
            f"Channel pair '{channel_pair}' not found. "
            f"Available pairs: {len(col_names)} total."
        )

    def _get_pairs_for_channel(
        self, channel: Union[str, List[str]], col_names: List[str]
    ) -> np.ndarray:
        """Get indices of all pairs involving the specified channel(s)."""
        if isinstance(channel, str):
            channels = [channel]
        else:
            channels = channel

        indices = []
        for idx, col in enumerate(col_names):
            pair = self._parse_channel_pair(col)
            if pair and (pair[0] in channels or pair[1] in channels):
                indices.append(idx)
        return np.array(indices)

    def load_events(self, tsv_path: Union[str, Path]):
        """Load events from TSV file. No parsing - just load as-is."""
        import pandas as pd

        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"Events file not found: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t")
        df.index.name = "epoch_idx"
        self._events_df = df
        return df

    @property
    def events(self):
        """Get loaded events DataFrame."""
        return self._events_df

    @property
    def has_events(self) -> bool:
        """Check if events are loaded."""
        return self._events_df is not None

    def get_connectivity_matrix(
        self,
        marker_name: str,
        epoch: Optional[int] = None,
        aggregation: str = "mean",
    ) -> np.ndarray:
        """Reconstruct full connectivity matrix from non-aggregated WSMI data."""
        info = self._load_marker(marker_name)
        if not info.get("is_connectivity_pairs", False):
            raise ValueError(
                f"Marker '{marker_name}' is not a non-aggregated connectivity marker. "
                "Use get_marker() for aggregated connectivity data."
            )

        data = info["data"]  # (epochs, channel_pairs)
        col_names = info["col_names"]
        unique_channels = info["channel_names_unique"]
        n_channels = len(unique_channels)

        # Create channel name to index mapping
        ch_to_idx = {ch: i for i, ch in enumerate(unique_channels)}

        # Get data for specified epoch or aggregate
        if epoch is not None:
            pair_values = data[epoch]
        else:
            pair_values = self._aggregate(data, aggregation, axis=0)

        # Reconstruct matrix
        matrix = np.zeros((n_channels, n_channels))
        for pair_idx, col in enumerate(col_names):
            pair = self._parse_channel_pair(col)
            if pair:
                i, j = ch_to_idx.get(pair[0]), ch_to_idx.get(pair[1])
                if i is not None and j is not None:
                    matrix[i, j] = pair_values[pair_idx]
                    matrix[j, i] = pair_values[pair_idx]  # Symmetric

        return matrix

    def get_connectivity_channels(self, marker_name: str) -> List[str]:
        """Get the unique channel names for a non-aggregated connectivity marker."""
        info = self._load_marker(marker_name)
        if not info.get("is_connectivity_pairs", False):
            raise ValueError(
                f"Marker '{marker_name}' is not a non-aggregated connectivity marker."
            )
        return info["channel_names_unique"]

    def get_channel_pair(
        self,
        marker_name: str,
        ch1: str,
        ch2: str,
        epoch: Optional[Union[int, List[int], slice]] = None,
    ) -> np.ndarray:
        """Get connectivity values for a specific channel pair."""
        return self.get_marker(
            marker_name, epoch=epoch, channel_pair=(ch1, ch2)
        )

    def to_dataframe(
        self,
        marker_name: str,
        band: Optional[str] = None,
        feature: Optional[str] = None,
        time_aggregation: Optional[str] = "mean",
        channel_aggregation: Optional[str] = None,
        include_events: bool = False,
    ):
        """Convert marker to DataFrame. Rows=epochs, Columns=channels (or aggregated)."""
        import pandas as pd

        info = self._load_marker(marker_name)
        marker_type = info["marker_type"]
        band_names = info["band_names"]
        feature_names = info["feature_names"]

        if (
            marker_type == "spectral"
            and band_names
            and len(band_names) > 1
            and band is None
        ):
            raise ValueError(
                f"Marker '{marker_name}' has {len(band_names)} bands. Specify band={band_names}"
            )
        if (
            marker_type == "sleep"
            and feature_names
            and len(feature_names) > 1
            and feature is None
        ):
            raise ValueError(
                f"Marker '{marker_name}' has {len(feature_names)} features. Specify feature={feature_names}"
            )

        data = self.get_marker(marker_name, band=band, feature=feature)

        # Handle ERP 3D data: (epochs, channels, timepoints)
        if marker_type == "erp" and data.ndim == 3:
            if time_aggregation is not None:
                # Aggregate over time dimension (axis=2)
                data = self._aggregate(data, time_aggregation, axis=2)
                # Now shape is (epochs, channels)
            else:
                # Flatten time into columns: each column is channel_timepoint
                n_epochs, n_channels, n_times = data.shape
                data = data.reshape(n_epochs, n_channels * n_times)
                # Update col_names to include time indices
                base_cols = info["col_names"] or [
                    f"ch_{i}" for i in range(n_channels)
                ]
                info["col_names"] = [
                    f"{ch}_t{t}" for ch in base_cols for t in range(n_times)
                ]

        if channel_aggregation is not None:
            agg_data = self._aggregate(data, channel_aggregation, axis=-1)
            col_name = f"{marker_name}_{channel_aggregation}"
            if band:
                col_name = f"{marker_name}_{band}_{channel_aggregation}"
            df = pd.DataFrame({col_name: agg_data})
        else:
            if data.ndim == 2:
                col_names = info["col_names"] or [
                    f"ch_{i}" for i in range(data.shape[1])
                ]
                df = pd.DataFrame(data, columns=col_names)
            elif data.ndim == 1:
                col_names = info["col_names"] or [
                    f"ch_{i}" for i in range(len(data))
                ]
                df = pd.DataFrame([data], columns=col_names)
            else:
                raise ValueError(
                    f"Cannot convert {data.ndim}D data. Specify band or feature first."
                )

        if (
            include_events
            and self._events_df is not None
            and len(df) == len(self._events_df)
        ):
            for col in self._events_df.columns:
                df[col] = self._events_df[col].values

        return df

    def _is_generic_channel_name(self, name: str) -> bool:
        """Check if a channel name is generic (e.g., ch_0, ch_1)."""
        if name.startswith("ch_") and name[3:].isdigit():
            return True
        return False

    def _has_generic_channels(self, channel_names: List[str]) -> bool:
        """Check if channel names are generic."""
        if not channel_names:
            return True
        return all(self._is_generic_channel_name(ch) for ch in channel_names)

    def _find_real_channel_names(self, n_channels: int) -> Optional[List[str]]:
        """Find real channel names from another marker with the same number of channels."""
        for marker_name in self.markers:
            try:
                info = self._load_marker(marker_name)
                col_names = info.get("col_names", [])
                # Check if this marker has real channel names with matching count
                if (
                    col_names
                    and len(col_names) == n_channels
                    and not self._has_generic_channels(col_names)
                ):
                    return col_names
            except Exception:
                continue
        return None

    def _aggregate(
        self, data: np.ndarray, method: str, axis: int
    ) -> np.ndarray:
        """Aggregate data using specified method."""
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
        if method == "sum":
            return np.sum(data, axis=axis)
        if method in ("trim_mean80", "trim_mean90"):
            from scipy import stats

            prop = 0.1 if method == "trim_mean80" else 0.05
            return stats.trim_mean(data, proportiontocut=prop, axis=axis)
        raise ValueError(f"Unknown aggregation: {method}")

    def plot_topomap(
        self,
        marker_name: str,
        epochs: Optional[Union[int, List[int], str]] = "mean",
        mne_info: Optional[Any] = None,  # mne.Info
        epochs_path: Optional[Union[str, Path]] = None,
        montage: str = "standard_1020",
        aggregation: str = "mean",
        cmap: str = "viridis",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """Plot topography map(s) for a marker."""
        try:
            from . import plotting_utils as pu
        except ImportError:
            import plotting_utils as pu

        pu.check_plotting_available()
        import matplotlib.pyplot as plt

        info = self.get_marker_info(marker_name)
        channel_names = info.channel_names
        n_channels = info.n_channels

        # Auto-detect and fix generic channel names
        if channel_names is None or self._has_generic_channels(channel_names):
            real_names = self._find_real_channel_names(n_channels)
            if real_names is not None:
                channel_names = real_names
            else:
                raise ValueError(
                    f"No real channel names available for marker '{marker_name}' "
                    f"and could not find them from other markers. "
                    "Cannot create topomap. Provide epochs_path or mne_info."
                )

        # Get or create MNE info
        if mne_info is not None:
            mne_info_to_use = mne_info
        elif epochs_path is not None:
            mne_info_to_use = pu.get_mne_info_from_epochs(epochs_path)
        else:
            mne_info_to_use = pu.create_mne_info_from_montage(
                channel_names, montage_name=montage
            )

        # Get data and prepare for plotting
        data = info.data

        # Build dictionary of {name: values_per_channel} for plotting
        topo_data = self._prepare_topo_data(
            marker_name, info, data, epochs, aggregation
        )

        # Create title
        if epochs == "mean":
            title_suffix = f"(mean over {info.n_epochs} epochs)"
        elif isinstance(epochs, int):
            title_suffix = f"(epoch {epochs})"
        else:
            title_suffix = f"({len(epochs)} epochs)"

        suptitle = f"{marker_name} {title_suffix}"

        # Plot
        fig = pu.plot_topomap_grid(
            topo_data,
            mne_info_to_use,
            channel_names,
            suptitle=suptitle,
            cmap=cmap,
            save_path=save_path,
        )

        if show:
            plt.show()

        return fig

    def _prepare_topo_data(
        self,
        marker_name: str,
        info,
        data: np.ndarray,
        epochs: Union[int, List[int], str],
        aggregation: str,
    ) -> Dict[str, np.ndarray]:
        """Prepare data dictionary for topomap plotting."""
        agg_func = np.nanmean if aggregation == "mean" else np.nanmedian
        topo_data = {}

        marker_type = info.marker_type

        if marker_type == "spectral" and data.ndim == 3:
            # Shape: (bands, epochs, channels)
            band_names = info.band_names or [
                f"band_{i}" for i in range(data.shape[0])
            ]

            for band_idx, band_name in enumerate(band_names):
                band_data = data[band_idx]  # (epochs, channels)
                if epochs == "mean":
                    values = agg_func(band_data, axis=0)
                elif isinstance(epochs, int):
                    values = band_data[epochs]
                else:
                    # Multiple epochs - create one entry per epoch per band
                    for ep in epochs:
                        topo_data[f"{band_name}_epoch{ep}"] = band_data[ep]
                    continue
                topo_data[band_name] = values

        elif marker_type == "sleep" and data.ndim == 3:
            # Shape: (features, epochs, channels)
            feature_names = info.feature_names or [
                f"feat_{i}" for i in range(data.shape[0])
            ]

            for feat_idx, feat_name in enumerate(feature_names):
                feat_data = data[feat_idx]  # (epochs, channels)
                if epochs == "mean":
                    values = agg_func(feat_data, axis=0)
                elif isinstance(epochs, int):
                    values = feat_data[epochs]
                else:
                    for ep in epochs:
                        topo_data[f"{feat_name}_epoch{ep}"] = feat_data[ep]
                    continue
                topo_data[feat_name] = values

        elif marker_type == "connectivity" and data.ndim == 3:
            # Shape: (epochs, channels_i, channels_j) - aggregate connectivity dimension
            if epochs == "mean":
                # Average over epochs, then over connectivity dimension
                epoch_agg = agg_func(data, axis=0)  # (channels_i, channels_j)
                values = agg_func(epoch_agg, axis=1)  # (channels,)
            elif isinstance(epochs, int):
                epoch_data = data[epochs]  # (channels_i, channels_j)
                values = agg_func(epoch_data, axis=1)  # (channels,)
                topo_data[f"epoch_{epochs}"] = values
            else:
                for ep in epochs:
                    epoch_data = data[ep]
                    values = agg_func(epoch_data, axis=1)
                    topo_data[f"epoch_{ep}"] = values
            if epochs == "mean":
                topo_data[marker_name] = values

        elif marker_type == "connectivity" and data.ndim == 2:
            # Already aggregated: (epochs, channels)
            if epochs == "mean":
                values = agg_func(data, axis=0)
            elif isinstance(epochs, int):
                values = data[epochs]
            else:
                for ep in epochs:
                    topo_data[f"epoch_{ep}"] = data[ep]
                return topo_data
            topo_data[marker_name] = values

        elif marker_type == "erp" and data.ndim == 3:
            # Shape: (epochs, channels, timepoints) - average over timepoints
            time_averaged = np.nanmean(data, axis=2)  # (epochs, channels)
            if epochs == "mean":
                values = agg_func(time_averaged, axis=0)
            elif isinstance(epochs, int):
                values = time_averaged[epochs]
            else:
                for ep in epochs:
                    topo_data[f"epoch_{ep}"] = time_averaged[ep]
                return topo_data
            topo_data[marker_name] = values

        elif data.ndim == 2:
            # Shape: (epochs, channels) - entropy, complexity, etc.
            if epochs == "mean":
                values = agg_func(data, axis=0)
            elif isinstance(epochs, int):
                values = data[epochs]
            else:
                for ep in epochs:
                    topo_data[f"epoch_{ep}"] = data[ep]
                return topo_data
            topo_data[marker_name] = values

        else:
            raise ValueError(
                f"Cannot create topomap for marker '{marker_name}' "
                f"with shape {data.shape}"
            )

        return topo_data

    def connectivity_matrix_to_dataframe(
        self,
        marker_name: str,
        epoch: Optional[int] = None,
        aggregation: str = "mean",
    ):
        """Convert non-aggregated connectivity to DataFrame with channel labels."""
        import pandas as pd

        matrix = self.get_connectivity_matrix(
            marker_name, epoch=epoch, aggregation=aggregation
        )
        channels = self.get_connectivity_channels(marker_name)
        return pd.DataFrame(matrix, index=channels, columns=channels)

    def summary(self) -> str:
        """Get a summary of all markers."""
        lines = [f"JuniferHDF5Reader: {self.path.name}", "=" * 50]
        for name in sorted(self.markers.keys()):
            info = self._load_marker(name)
            shape_str = str(info["data"].shape)
            extra = ""
            if info.get("is_connectivity_pairs", False):
                n_ch = (
                    len(info["channel_names_unique"])
                    if info["channel_names_unique"]
                    else "?"
                )
                extra = f" [non-aggregated, {n_ch} channels]"
            lines.append(
                f"  {name}: {info['marker_type']}, shape={shape_str}{extra}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"JuniferHDF5Reader('{self.path}', markers={len(self.markers)})"

    def __str__(self):
        return self.summary()


# Convenience function
def read_junifer_h5(path: Union[str, Path]) -> JuniferHDF5Reader:
    """Convenience function to create a JuniferHDF5Reader."""
    return JuniferHDF5Reader(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python junifer_h5_reader.py <h5_file>")
        sys.exit(1)
    reader = JuniferHDF5Reader(sys.argv[1])
    print(reader.summary())
