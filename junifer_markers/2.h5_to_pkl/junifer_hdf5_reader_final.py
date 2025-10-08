#!/usr/bin/env python3
"""
Final Junifer HDF5 Reader

This module provides the definitive interface for reading junifer HDF5 feature files
using junifer's built-in HDF5FeatureStorage class. This is the recommended approach
for reading junifer-generated HDF5 files.

Author: Generated for junifer_eeg project
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

try:
    from junifer.storage import HDF5FeatureStorage
except ImportError:
    print(
        "Error: junifer not found. Please install junifer or ensure it's in your Python path."
    )
    sys.exit(1)


class JuniferHDF5Reader:
    """Final reader for junifer HDF5 feature storage files.

    This class uses junifer's built-in HDF5FeatureStorage to read features,
    ensuring compatibility with junifer's storage format and handling all
    the MD5 hashing and metadata management automatically.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the junifer HDF5 file

    Examples
    --------
    >>> reader = JuniferHDF5Reader("icm_complete_features.h5")
    >>> features = reader.list_features()
    >>> alpha_data = reader.read_feature("EEG_per_channel_alpha_spectralpower")
    >>> spectral_bands = reader.extract_spectral_features()
    """

    def __init__(self, hdf5_path: Union[str, Path]):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Initialize junifer storage
        self.storage = HDF5FeatureStorage(str(self.hdf5_path))

        # Cache for feature list
        self._features_cache = None

    def _get_features(self) -> Dict[str, Dict[str, Any]]:
        """Get cached feature list."""
        if self._features_cache is None:
            self._features_cache = self.storage.list_features()
        return self._features_cache

    def list_features(
        self, detailed: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """List all available features in the HDF5 file.

        Parameters
        ----------
        detailed : bool, optional
            If True, return detailed metadata for each feature.
            If False, return only feature names (default False).

        Returns
        -------
        list of str or dict
            Feature names or detailed metadata dictionary.
        """
        features = self._get_features()

        if detailed:
            return {
                meta["name"]: {
                    "md5": md5_key,
                    "type": meta.get("type", "unknown"),
                    "marker": meta.get("marker", {}),
                    "element_keys": meta.get("_element_keys", []),
                }
                for md5_key, meta in features.items()
            }
        else:
            return [meta["name"] for meta in features.values()]

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature to get info for.

        Returns
        -------
        dict
            Feature metadata and information.
        """
        features = self._get_features()

        # Find the feature by name
        for md5_key, meta in features.items():
            if meta.get("name") == feature_name:
                return {"name": feature_name, "md5": md5_key, "metadata": meta}

        available = [meta["name"] for meta in features.values()]
        raise ValueError(
            f"Feature '{feature_name}' not found. Available: {available}"
        )

    def read_feature(self, feature_name: str) -> Dict[str, Any]:
        """Read a feature from the HDF5 file using junifer's interface.

        Parameters
        ----------
        feature_name : str
            Name of the feature to read.

        Returns
        -------
        dict
            Feature data dictionary containing:
            - 'data': numpy array with feature values
            - 'kind': storage type (vector, matrix, timeseries, etc.)
            - 'column_headers': column/channel names
            - 'element': subject/session information
            - Additional metadata depending on storage type
        """
        try:
            return self.storage.read(feature_name=feature_name)
        except Exception as e:
            available = self.list_features()
            raise ValueError(
                f"Failed to read feature '{feature_name}': {e}. "
                f"Available features: {available}"
            )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the HDF5 file contents.

        Returns
        -------
        dict
            Summary statistics including feature counts by type, 
            data shapes, etc.
        """
        features = self._get_features()

        stats = {
            "total_features": len(features),
            "feature_types": {},
            "storage_kinds": {},
            "marker_types": {},
            "feature_names": [],
        }

        for md5_key, meta in features.items():
            feature_name = meta.get("name", "unknown")
            stats["feature_names"].append(feature_name)

            # Count by storage type
            storage_type = meta.get("type", "unknown")
            stats["storage_kinds"][storage_type] = (
                stats["storage_kinds"].get(storage_type, 0) + 1
            )

            # Count by marker type
            marker_info = meta.get("marker", {})
            if isinstance(marker_info, dict):
                marker_name = marker_info.get("name", "unknown")
            else:
                marker_name = "unknown"
            stats["marker_types"][marker_name] = (
                stats["marker_types"].get(marker_name, 0) + 1
            )

            # Categorize by feature type
            if "spectralpower" in feature_name:
                category = "spectral"
            elif "symbolicmutualinformation" in feature_name:
                category = "connectivity"
            elif "topography" in feature_name or "contrast" in feature_name:
                category = "erp"
            elif "cnv" in feature_name:
                category = "cnv"
            elif "decoding" in feature_name:
                category = "decoding"
            elif "kolmogorov" in feature_name or "permutation" in feature_name:
                category = "complexity"
            elif "psdsummary" in feature_name:
                category = "summary"
            else:
                category = "other"

            stats["feature_types"][category] = (
                stats["feature_types"].get(category, 0) + 1
            )

        return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python junifer_hdf5_reader_final.py <hdf5_file>")
        sys.exit(1)

    hdf5_file = sys.argv[1]

    try:
        reader = JuniferHDF5Reader(hdf5_file)
        print("=== Junifer HDF5 File Analysis ===")
        print(f"File: {hdf5_file}")

        # Summary statistics
        stats = reader.get_summary_statistics()
        print("\nSummary:")
        print(f"  Total features: {stats['total_features']}")
        print(f"  Feature categories: {stats['feature_types']}")
        print(f"  Storage types: {stats['storage_kinds']}")
        print(f"  Marker types: {stats['marker_types']}")

        # List all features
        print("\nAvailable features:")
        features = reader.list_features()
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
