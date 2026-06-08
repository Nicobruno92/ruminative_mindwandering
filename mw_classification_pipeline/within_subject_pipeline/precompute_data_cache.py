#!/usr/bin/env python
"""
Pre-compute and cache processed data for all (contrast × family) combinations.

Each job currently spends ~5 minutes loading 784k rows of CSVs and filtering
features before doing ~43 seconds of actual ML. This script runs that loading
once per combination and saves compact pickle files (~4 MB each) that jobs
can read in < 1 second.

USAGE (run once before launching jobs):
    conda activate ML
    python precompute_data_cache.py --config config.yaml

OUTPUT: results/data_cache/{contrast}__{family}.pkl  (one per combination)
"""

import os
import sys
import argparse
import pickle
import yaml
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import load_config, prepare_data_for_contrast, get_project_root


def _filter_features_by_family(X, prefixes):
    if prefixes is None:
        return X

    def _matches(col, prefix):
        p = re.escape(prefix)
        return bool(re.search(rf"_{p}(_|$)", col)) or bool(re.match(rf"{p}(_|$)", col))

    selected = [c for c in X.columns if any(_matches(c, p) for p in prefixes)]
    return X[selected]


def cache_path_for(cache_dir: str, contrast: str, family: str) -> str:
    safe = contrast.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}__{family}.pkl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cache_dir", default=None,
                   help="Where to write cache files (default: results/data_cache/)")
    args = p.parse_args()

    config = load_config(args.config)

    contrasts = config.get("run_contrasts", [])
    families_cfg = config.get("feature_families", {})
    run_families = config.get("run_families", ["all"])

    project_root = get_project_root()
    results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    if not os.path.isabs(results_root):
        results_root = str(project_root / results_root)

    cache_dir = args.cache_dir or os.path.join(results_root, "data_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Cache directory: {cache_dir}")
    print(f"Contrasts: {contrasts}")
    print(f"Families:  {run_families}")
    print()

    # Inject epoch_types for data loading
    for family in run_families:
        fam_cfg = families_cfg.get(family, {})
        epoch_types = fam_cfg.get("epoch_types", ["state"]) if isinstance(fam_cfg, dict) else ["state"]
        prefixes = fam_cfg.get("prefixes") if isinstance(fam_cfg, dict) else None

        for contrast in contrasts:
            out = cache_path_for(cache_dir, contrast, family)
            if os.path.exists(out):
                size_mb = os.path.getsize(out) / 1e6
                print(f"  SKIP {contrast} / {family}  (already cached, {size_mb:.1f} MB)")
                continue

            print(f"  Loading {contrast} / {family} ...", flush=True)
            t0 = time.time()

            config["epoch_types"] = epoch_types
            df, X, y, groups, feature_cols = prepare_data_for_contrast(
                config, contrast, verbose=False
            )
            X = _filter_features_by_family(X, prefixes)
            feature_cols = X.columns.tolist()

            payload = {
                "df": df,
                "X": X,
                "y": y,
                "groups": groups,
                "feature_cols": feature_cols,
                "contrast": contrast,
                "family": family,
                "epoch_types": epoch_types,
            }

            with open(out, "wb") as f:
                pickle.dump(payload, f, protocol=5)

            elapsed = time.time() - t0
            size_mb = os.path.getsize(out) / 1e6
            print(f"  OK  {contrast} / {family}  "
                  f"({len(X)} samples × {len(feature_cols)} features, "
                  f"{size_mb:.1f} MB, {elapsed:.1f}s)")

    print(f"\nDone. {len(contrasts) * len(run_families)} cache files in {cache_dir}")


if __name__ == "__main__":
    main()
