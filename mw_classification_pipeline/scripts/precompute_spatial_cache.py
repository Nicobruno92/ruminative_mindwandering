#!/usr/bin/env python
"""
Pre-build the per-channel data caches for spatial decoding (run ONCE, sequentially).

The spatial SLURM arrays (true + permutation) each read a per-channel cache keyed by
(contrast, family, data_format). Building it here first avoids a "thundering herd"
where many array tasks cache-miss simultaneously and race on the same pickle.

Works for both the LOSO config (results_root under data_paths) and the within-subject
config (results_dir under project). Caches one pickle per (contrast × family).

USAGE:
    python precompute_spatial_cache.py --config <loso|ws>/spatial_decoding/config.yaml
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import load_config, get_project_root
from utils.spatial_decoding_utils import load_or_prepare_data


def resolve_results_root(config: dict) -> str:
    root = (config.get("data_paths", {}).get("results_root")
            or config.get("project", {}).get("results_dir")
            or "results/MW_Classification")
    if not os.path.isabs(root):
        root = str(get_project_root() / root)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    contrasts = config["spatial_decoding"].get("contrasts", config.get("run_contrasts", []))
    families = config.get("run_families", [config.get("active_family") or "all"])
    cache_dir = os.path.join(resolve_results_root(config), "data_cache")

    print(f"Cache dir: {cache_dir}")
    print(f"Contrasts: {contrasts}\nFamilies: {families}\n")

    for family in families:
        fam_cfg = config["feature_families"][family]
        config["epoch_types"] = fam_cfg["epoch_types"]
        prefixes = fam_cfg.get("prefixes")
        for contrast in contrasts:
            print(f"  building cache: {contrast} / {family} ...", flush=True)
            df, X, y, groups, cols = load_or_prepare_data(
                config, contrast, family, prefixes, cache_dir, verbose=False
            )
            print(f"    OK ({len(X)} samples × {len(cols)} features)")

    print("\nDone. Caches ready for the spatial SLURM arrays.")


if __name__ == "__main__":
    main()
