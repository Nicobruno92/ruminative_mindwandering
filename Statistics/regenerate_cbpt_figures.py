"""
Regenerate CBPT result figures (PNG + SVG) from saved pickles.

This script re-creates the per-marker visualisation report for a cluster-based
permutation test (CBPT) model directory **without re-running the permutation
test itself**. It reads the ``results.pkl`` written by the pipeline for each
marker and replays :func:`plot_results.create_results_report`, which now emits
both PNG and SVG by default (see ``plot_results.save_figure_multiformat``).

Use it to add SVG versions of already-computed figures, or to refresh figures
after a plotting-code change, while reusing the expensive CBPT statistics on
disk.

Usage
-----
    python Statistics/regenerate_cbpt_figures.py <model_dir>
    python Statistics/regenerate_cbpt_figures.py <model_dir> --config CONFIG

Example
-------
    python Statistics/regenerate_cbpt_figures.py \
        results/andrillon_cluster/onoff_valence_selfother_time_time_on_task_confidence__target_onoff
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import yaml

from plot_results import create_results_report, plot_cluster_topomap, save_figure_multiformat


# =============================================================================
# Helper functions
# =============================================================================
def find_marker_results(model_dir: Path) -> List[Path]:
    """
    Locate every marker ``results.pkl`` under a CBPT model directory.

    Parameters
    ----------
    model_dir : pathlib.Path
        Model directory containing one subdirectory per marker.

    Returns
    -------
    list of pathlib.Path
        Sorted list of ``results.pkl`` paths, one per marker subdirectory.
    """
    return sorted(
        marker_dir / "results.pkl"
        for marker_dir in model_dir.iterdir()
        if marker_dir.is_dir() and (marker_dir / "results.pkl").exists()
    )


def regenerate_marker_figures(results_pkl: Path, verbose: bool = True) -> None:
    """
    Regenerate the figure set for a single marker from its saved results.

    Replays the standard report (topographic map, cluster details, t-statistic
    distribution) in PNG + SVG. If the pickle also carries MCC-corrected cluster
    p-values, the corrected topomap is regenerated as well, mirroring
    ``apply_mcc_postprocessing``.

    Parameters
    ----------
    results_pkl : pathlib.Path
        Path to a marker's ``results.pkl``.
    verbose : bool, default True
        Print per-marker progress.
    """
    with open(results_pkl, "rb") as handle:
        result: Dict = pickle.load(handle)

    output_dir = results_pkl.parent
    marker_name = result["marker_name"]

    if verbose:
        print(f"→ {marker_name}  ({output_dir.name})")

    create_results_report(
        t_stats=result["t_stats"],
        clusters=result["clusters"],
        cluster_stats=result["cluster_stats"],
        cluster_p_values=result["cluster_p_values"],
        info=result["info"],
        threshold=result["threshold"],
        alpha=result["alpha"],
        marker_name=marker_name,
        output_dir=str(output_dir),
        config=result.get("config"),
    )

    # If MCC correction is present, also regenerate the corrected topomap.
    if "cluster_p_values_corrected" in result:
        safe_marker_name = marker_name.replace("/", "_").replace(" ", "_")
        corrected_path = output_dir / f"results_{safe_marker_name}_topomap_corrected.png"
        mcc_alpha = result.get("correction_alpha", result["alpha"])
        plot_cluster_topomap(
            t_stats=result["t_stats"],
            clusters=result["clusters"],
            cluster_p_values=result["cluster_p_values_corrected"],
            info=result["info"],
            alpha=mcc_alpha,
            title=f"Spatial Cluster Test - {marker_name} (MCC Applied)",
            save_path=str(corrected_path),
            cluster_p_values_uncorrected=result["cluster_p_values"],
        )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Parse arguments and regenerate figures for all markers in a model dir."""
    parser = argparse.ArgumentParser(
        description="Regenerate CBPT result figures (PNG + SVG) from saved pickles "
        "without re-running the permutation test."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to a CBPT model directory containing marker subdirectories.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics/config.yaml",
        help="Path to configuration file (used only to confirm availability).",
    )
    args = parser.parse_args()

    # Load config so the run fails loudly if the expected file is missing,
    # keeping parity with the rest of the pipeline (no silent fallbacks).
    with open(args.config) as handle:
        yaml.safe_load(handle)

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_dir}")

    results_pkls = find_marker_results(model_dir)
    if not results_pkls:
        raise FileNotFoundError(f"No marker results.pkl found under {model_dir}")

    print("=" * 80)
    print(f"Regenerating figures for {len(results_pkls)} markers in {model_dir}")
    print("=" * 80)

    for results_pkl in results_pkls:
        regenerate_marker_figures(results_pkl)

    print("=" * 80)
    print(f"✓ Done. Regenerated PNG + SVG for {len(results_pkls)} markers.")
    print("=" * 80)


if __name__ == "__main__":
    main()
