"""Classify I/E intervention type (Inclusion vs Exclusion) using LOSO.

Features are aggregated per subject × block. Only intervention blocks
(inclusion, exclusion) are used; baseline blocks are excluded.

Usage:
    conda activate ML
    cd <repo_root>
    python Behavior/Classification/run_ie_classification.py \
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
    build_block_level_features,
    build_model,
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
    """Run LOSO I/E intervention classification for all configured models.

    Parameters
    ----------
    config_path : path to config.yaml
    """
    config = yaml.safe_load(open(config_path))
    feature_cols = get_feature_cols(config)
    df = merge_all_features(config)

    # Filter to intervention blocks only
    exclude = config["ie_classification"]["exclude_conditions"]
    df_ie = df[~df["inclusion_exclusion"].isin(exclude)].copy()
    ie_counts = df_ie["inclusion_exclusion"].value_counts().to_dict()
    print(f"[I/E] Rows after filtering baseline: {len(df_ie)} ({ie_counts})")

    target_col = config["ie_classification"]["target_col"]
    positive_class = config["ie_classification"]["positive_class"]
    results_root = Path(config["data"]["results_root"])

    X, y, groups = build_block_level_features(df_ie, feature_cols, target_col, positive_class)
    print(f"[I/E] X={X.shape}, y dist={_unique_counts(y)}")

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
            scope=config["permutation"]["scope_ie"],
            n_jobs=config["loso"]["n_jobs"],
        )

        output_dir = results_root / "ie_intervention" / model_type
        save_results(results, perm_results, feature_cols, output_dir, config)
        plot_results(results, perm_results, output_dir, feature_names=feature_cols)
        print(f"    Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I/E intervention classification (Inclusion vs Exclusion)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
