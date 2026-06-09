"""Classify patient groups (Controls vs Risk of Depression) using LOSO.

Usage:
    conda activate ML
    cd <repo_root>
    python Behavior/Classification/run_group_classification.py \
        --config Behavior/Classification/config.yaml
"""
# === Imports ===
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_model,
    build_probe_level_features,
    build_subject_level_features,
    get_feature_cols,
    merge_all_features,
    plot_results,
    run_loso,
    run_permutations,
    save_results,
)


# === Helpers ===

def _unique_counts(y: np.ndarray) -> dict:
    """Return {label: count} dict for display."""
    vals, counts = np.unique(y, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


# === Main ===

def main(config_path: str) -> None:
    """Run LOSO group classification for all configured modes and models.

    Parameters
    ----------
    config_path : path to config.yaml
    """
    config = yaml.safe_load(open(config_path))
    feature_cols = get_feature_cols(config)
    df = merge_all_features(config)

    target_col = config["group_classification"]["target_col"]
    positive_class = config["group_classification"]["positive_class"]
    results_root = Path(config["data"]["results_root"])

    modes = list(config["group_classification"]["modes"])
    if config["group_classification"].get("run_probe_level", False) and "probe_level" not in modes:
        modes = modes + ["probe_level"]

    for mode in modes:
        if mode == "subject_level":
            X, y, groups = build_subject_level_features(df, feature_cols, target_col, positive_class)
        elif mode == "probe_level":
            X, y, groups = build_probe_level_features(df, feature_cols, target_col, positive_class)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        print(f"\n[Group | {mode}] X={X.shape}, y dist={_unique_counts(y)}")

        for model_type in config["models"]["types"]:
            print(f"  Model: {model_type}")
            model = build_model(model_type, config)
            results = run_loso(X, y, groups, model, n_jobs=config["loso"]["n_jobs"])
            print(f"    AUC={results['auc']:.3f}  BACC={results['balanced_accuracy']:.3f}")

            perm_results = run_permutations(
                X, y, groups,
                build_model(model_type, config),
                n_perms=config["permutation"]["n_perms"],
                random_state=config["permutation"]["random_state"],
                scope=config["permutation"]["scope_group"],
                n_jobs=config["loso"]["n_jobs"],
            )

            output_dir = results_root / "group" / mode / model_type
            save_results(results, perm_results, feature_cols, output_dir, config)
            plot_results(results, perm_results, output_dir, feature_names=feature_cols)
            print(f"    Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group classification (Controls vs Risk of Depression)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
