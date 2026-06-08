#!/usr/bin/env python
"""
Cross-decoding generalization matrix — entry point (predictions-based).

Assembles the matrix POST-HOC from the per-probe predictions each per-dimension
classification run already saves (``*_consolidated_sample_predictions.csv``).
Nothing is retrained, no data is reloaded, and the per-dimension classifications
are not modified.

Cell (model M -> dimension D) = AUC of M's saved out-of-fold scores
(``proba_mean``) against D's binary labels (``y_true_first``), over the probes
present in BOTH runs. The diagonal reproduces each dimension's own AUC.

USAGE (after the per-dimension classifications have been run):
    python run_cross_decoding.py --config config.yaml
    python run_cross_decoding.py --config config.yaml --scheme LOSO
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

# mw_classification_pipeline/ on path for utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import load_config, get_project_root
from utils.cross_decoding_utils import (
    cross_decode_from_predictions,
    save_cross_decoding_outputs,
)


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cross-decoding generalization matrix.")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--scheme", type=str, default=None,
                        help="Override config schemes; assemble a single scheme "
                             "(e.g. WithinSubject or LOSO).")
    return parser.parse_args()


# =============================================================================
# Prediction loading
# =============================================================================
def _consolidate_true_runs(true_runs_dir: str) -> pd.DataFrame:
    """Build per-probe predictions by averaging across ``true_runs/run_*/`` files.

    Used when a run did not write a consolidated file (e.g. LOSO SLURM jobs leave
    only per-run sample predictions). Averages ``y_proba`` per probe and takes the
    (constant) ``y_true`` as ``y_true_first``, matching the consolidated schema.
    """
    run_files = sorted(glob.glob(
        os.path.join(true_runs_dir, "run_*", "*_sample_predictions.csv")))
    if not run_files:
        raise FileNotFoundError(f"No per-run sample predictions under {true_runs_dir}")
    stacked = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
    id_cols = [c for c in ["subject", "task", "probe_number"] if c in stacked.columns]
    summary = stacked.groupby(id_cols).agg(
        y_true_first=("y_true", "first"),
        proba_mean=("y_proba", "mean"),
    ).reset_index()
    return summary


def load_prediction_table(
    results_root: str, scheme: str, contrast_folder: str, family: str, model: str,
    predictions_glob: str, exclude_substring: str,
) -> pd.DataFrame:
    """Load per-probe predictions for one dimension.

    Prefers a written ``*_consolidated_sample_predictions.csv``; if none exists,
    consolidates on the fly from ``true_runs/run_*/`` (the LOSO case).

    Parameters
    ----------
    results_root, scheme, contrast_folder, family, model : str
        Locate ``{results_root}/{scheme}/{contrast_folder}/{family}/{model}/``.
    predictions_glob : str
        Glob for the consolidated predictions file inside that folder.
    exclude_substring : str
        Skip files whose name contains this substring (e.g. permutation files).

    Returns
    -------
    pd.DataFrame
        Per-probe predictions with ``y_true_first`` and ``proba_mean``.
    """
    folder = os.path.join(results_root, scheme, contrast_folder, family, model)
    matches = [
        f for f in sorted(glob.glob(os.path.join(folder, predictions_glob)))
        if exclude_substring not in os.path.basename(f)
    ]
    if matches:
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous predictions in {folder}: {[os.path.basename(m) for m in matches]}")
        return pd.read_csv(matches[0])

    true_runs_dir = os.path.join(folder, "true_runs")
    if os.path.isdir(true_runs_dir):
        return _consolidate_true_runs(true_runs_dir)

    raise FileNotFoundError(
        f"No consolidated predictions (or true_runs/) in {folder}. Run the "
        f"per-dimension classification with save_probabilities first."
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Assemble the cross-decoding matrix for the configured scheme(s)."""
    args = parse_args()
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    cd = config["cross_decoding"]

    results_root = cd["results_root"]
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)

    schemes = [args.scheme] if args.scheme else cd.get("schemes", ["WithinSubject"])
    family = cd.get("family", "all")
    model = cd.get("model", "rf")
    min_overlap = cd.get("min_overlap", 30)
    dimensions = cd["dimensions"]

    overrides = cd.get("scheme_folder_overrides", {})

    for scheme in schemes:
        print(f"\n{'='*60}\nAssembling {scheme} cross-decoding matrix\n{'='*60}")
        scheme_dims = {**dimensions, **overrides.get(scheme, {})}
        pred_tables = {
            dim: load_prediction_table(
                results_root, scheme, contrast_folder, family, model,
                cd.get("predictions_glob", "*consolidated_sample_predictions.csv"),
                cd.get("exclude_substring", "permutation"),
            )
            for dim, contrast_folder in scheme_dims.items()
        }
        for dim, table in pred_tables.items():
            print(f"  {dim:11s}: {len(table)} probes")

        result = cross_decode_from_predictions(pred_tables, min_overlap=min_overlap)

        out_dir = os.path.join(results_root, "cross_decoding", f"{scheme}_{model}_{family}")
        save_cross_decoding_outputs(
            result, out_dir,
            title=f"Cross-decoding AUC ({scheme}, {model.upper()})",
        )
        print(f"\n{scheme} AUC matrix (train rows × test cols):")
        print(result["mean"].round(3).to_string())
        print(f"\nProbe overlap per cell:")
        print(result["n_overlap"].to_string())
        print(f"\nResults → {out_dir}")


if __name__ == "__main__":
    main()
