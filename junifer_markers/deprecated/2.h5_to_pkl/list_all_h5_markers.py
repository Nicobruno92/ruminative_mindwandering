#!/usr/bin/env python3
"""List all markers with their names and structures from H5."""

import sys

sys.path.insert(0, "/home/gio/Documents/Repos/junifer_eeg")
from junifer.storage import HDF5FeatureStorage

HDF5_PATH = "explore_spectral.h5"

# Known markers from the YAML config
expected_markers = [
    "EEG_psd_bands_spectralpower",
    "EEG_psd_relative_spectralpower",
    "EEG_wsmi_theta_symbolicmutualinformation",
    "EEG_wsmi_alpha_symbolicmutualinformation",
    "EEG_wsmi_beta_symbolicmutualinformation",
    "EEG_wsmi_gamma_symbolicmutualinformation",
    "EEG_PE_theta_permutationentropy",
    "EEG_PE_alpha_permutationentropy",
    "EEG_PE_beta_permutationentropy",
    "EEG_PE_gamma_permutationentropy",
    "EEG_kolmogorov_complexity_kolmogorovcomplexity",
    "EEG_P1_timelockedtopo",
    "EEG_N1_timelockedtopo",
    "EEG_P2_timelockedtopo",
    "EEG_P3a_timelockedtopo",
    "EEG_P3b_timelockedtopo",
]

storage = HDF5FeatureStorage(HDF5_PATH)

print("=" * 80)
print("CHECKING ALL EXPECTED MARKERS")
print("=" * 80)

found = []
missing = []

for marker_name in expected_markers:
    try:
        data = storage.read(feature_name=marker_name)
        shape = data["data"].shape
        n_cols = len(data.get("column_headers", []))

        # Detect type
        if "wsmi" in marker_name.lower():
            mtype = "🔗 WSMI"
        elif "pe_" in marker_name.lower():
            mtype = "📊 PE"
        elif "kolmogorov" in marker_name.lower():
            mtype = "🧮 Kolmogorov"
        elif "psd" in marker_name.lower():
            mtype = "📈 Spectral"
        elif any(x in marker_name.lower() for x in ["p1", "n1", "p2", "p3"]):
            mtype = "⚡ ERP"
        else:
            mtype = "❓ Unknown"

        print(f"\n✅ {mtype} {marker_name}")
        print(f"   Data shape: {shape}")
        print(f"   Columns: {n_cols}")
        if n_cols > 0 and n_cols <= 10:
            print(f"   Col headers: {data['column_headers']}")
        elif n_cols > 0:
            print(f"   First 3 cols: {data['column_headers'][:3]}")
            print(f"   Last 3 cols: {data['column_headers'][-3:]}")

        found.append(marker_name)
    except Exception as e:
        print(f"\n❌ {marker_name}")
        print(f"   Error: {e}")
        missing.append(marker_name)

print("\n" + "=" * 80)
print(f"SUMMARY: {len(found)}/{len(expected_markers)} markers found")
print("=" * 80)

if missing:
    print(f"\nMissing markers ({len(missing)}):")
    for m in missing:
        print(f"  - {m}")
