"""
Family-level (omnibus) permutation test for the Andrillon cluster pipeline.

WHAT QUESTION THIS ANSWERS
--------------------------
The marker-wise FDR correction asks, for each marker: "does *this* marker track
the probe dimension, after accounting for the other markers tested?" It has to
name a culprit, which needs strong evidence concentrated in one marker.

This module asks a different, equally legitimate question: "does *any* marker in
the family track the dimension?" A dimension whose signal is spread thinly over
six moderate markers can fail every marker-wise test while being, collectively,
far from chance. Reporting only the marker-wise result would call such a
dimension null, which is a claim the data does not support.

The two are complementary and must be reported together: the omnibus says
whether there is signal, the marker-wise says where it is.

WHY A PERMUTATION TEST AND NOT A BINOMIAL
-----------------------------------------
Counting markers with p <= alpha and comparing against Binomial(m, alpha)
assumes the markers are independent. They are not — psd_relative_* are
compositional and the ratios share denominators with their components, so
markers co-fluctuate. Here the null is built from the permutations themselves:
permutation k is the *same* shuffle of the underlying data for every marker, so
the joint behaviour of the family under the null — including its dependence
structure — is reproduced exactly, with no distributional assumption.

This requires ``null_cluster_stats`` in each marker's results.pkl, written by
``andrillon_pipeline.py``. Runs predating that are not usable: the null cluster
statistics cannot be reconstructed without re-fitting every permuted LMM.

TWO STATISTICS ARE REPORTED
---------------------------
count   Number of markers whose own permutation p-value is <= alpha. Sensitive
        to signal spread thinly across many markers.
min_p   Smallest marker p-value in the family (Tippett). Its permutation
        p-value is a FWER-corrected significance for the single best marker,
        valid under arbitrary dependence — an assumption-free alternative to
        the Benjamini-Hochberg correction, which requires positive dependence.

Usage
-----
    python Stats_andrillon/omnibus_test.py --config Stats_andrillon/config_andrillon.yaml
    python Stats_andrillon/omnibus_test.py --config ... --model-dir <one dir>
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Statistics"))

from apply_mcc_postprocessing import (  # noqa: E402
    load_marker_results,
    partition_results_by_config,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

NULL_KEY = "null_cluster_stats"
OUTPUT_NAME = "omnibus_test.csv"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_null_matrices(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Stack the per-permutation null cluster statistics into (markers x perms).

    Parameters
    ----------
    results : List[Dict]
        Marker results belonging to one correction family.

    Returns
    -------
    pos : np.ndarray
        Maximum positive cluster statistic, shape (n_markers, n_permutations).
    neg : np.ndarray
        Minimum negative cluster statistic, same shape.
    names : List[str]
        Marker names, row-aligned with ``pos`` and ``neg``.

    Raises
    ------
    ValueError
        If any marker lacks the null distributions, or if markers disagree on
        the number of permutations. Both are fatal: the omnibus null is only
        valid if permutation k means the same data shuffle for every marker,
        and a differing permutation count is direct evidence that it does not.
    """
    missing = [r.get("marker_name", "?") for r in results if NULL_KEY not in r]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(results)} marker(s) have no '{NULL_KEY}'. This "
            f"analysis needs the per-permutation null cluster statistics, which "
            f"are written by andrillon_pipeline.py and cannot be reconstructed "
            f"afterwards. Re-run the pipeline for these markers.\n"
            f"Missing e.g.: {missing[:5]}"
        )

    names, pos_rows, neg_rows = [], [], []
    for result in sorted(results, key=lambda r: r["marker_name"]):
        nulls = result[NULL_KEY]
        if "pos_max" not in nulls:
            raise ValueError(
                f"{result['marker_name']}: expected separate-sign nulls "
                f"('pos_max'/'neg_min'), found {sorted(nulls)}. The omnibus test "
                f"assumes the pipeline ran with separate_signs=True."
            )
        names.append(result["marker_name"])
        pos_rows.append(np.asarray(nulls["pos_max"], dtype=float))
        neg_rows.append(np.asarray(nulls["neg_min"], dtype=float))

    lengths = {len(r) for r in pos_rows} | {len(r) for r in neg_rows}
    if len(lengths) != 1:
        raise ValueError(
            f"Markers disagree on the number of permutations: {sorted(lengths)}. "
            f"The omnibus null requires permutation k to be the same shuffle "
            f"across every marker in the family."
        )

    return np.vstack(pos_rows), np.vstack(neg_rows), names


def observed_marker_pvalues(results: List[Dict], names: List[str]) -> np.ndarray:
    """
    Return each marker's observed p-value, row-aligned with ``names``.

    A marker is represented by its most extreme cluster (the max-statistic
    cluster), matching the ``unit="marker"`` convention of the FDR correction.
    A marker with no cluster produced no evidence and is represented by p = 1.

    Parameters
    ----------
    results : List[Dict]
        Marker results for the family.
    names : List[str]
        Marker names defining the row order.

    Returns
    -------
    np.ndarray
        Observed p-value per marker.
    """
    by_name = {r["marker_name"]: r for r in results}
    out = np.ones(len(names), dtype=float)
    for i, name in enumerate(names):
        p = np.asarray(by_name[name].get("cluster_p_values", []), dtype=float)
        p = p[~np.isnan(p)]
        if p.size:
            out[i] = float(p.min())
    return out


def permutation_marker_pvalues(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """
    Compute each permutation's marker-level p-value, leave-one-out.

    Permutation k is treated as if it were the observed data: its cluster
    statistic is ranked against the null formed by the *other* permutations.
    Excluding k from its own reference set is what keeps the resulting omnibus
    p-value calibrated — including it would bias every permutation towards
    looking less extreme than the true observed data, which is compared against
    the full set.

    Parameters
    ----------
    pos : np.ndarray
        Positive-cluster nulls, shape (n_markers, n_permutations).
    neg : np.ndarray
        Negative-cluster nulls, same shape.

    Returns
    -------
    np.ndarray
        p-values, shape (n_markers, n_permutations).
    """
    n_markers, n_perm = pos.shape

    # ge_pos[i, k] = #{j : pos[i, j] >= pos[i, k]}, ties included.
    # Counting "greater or equal" via searchsorted needs an ascending array, so
    # negate: {pos[j] >= pos[k]} is exactly {-pos[j] <= -pos[k]}.
    ge_pos = np.array(
        [np.searchsorted(np.sort(-pos[i]), -pos[i], side="right")
         for i in range(n_markers)],
        dtype=float,
    )
    # le_neg[i, k] = #{j : neg[i, j] <= neg[i, k]}, ties included.
    le_neg = np.array(
        [np.searchsorted(np.sort(neg[i]), neg[i], side="right")
         for i in range(n_markers)],
        dtype=float,
    )

    # Leave-one-out. The self-comparison (j == k) is always a hit, so removing
    # it leaves (count - 1) hits out of (n_perm - 1) references; the
    # Phipson & Smyth +1 on both then cancels back to count / n_perm.
    p_pos = ge_pos / n_perm
    p_neg = le_neg / n_perm

    return np.minimum(np.minimum(p_pos, p_neg), 1.0)


def run_omnibus(
    results: List[Dict],
    alpha: float,
) -> Dict[str, float]:
    """
    Run both omnibus statistics for one correction family.

    Parameters
    ----------
    results : List[Dict]
        Marker results for the family.
    alpha : float
        Threshold defining a "significant" marker for the count statistic.

    Returns
    -------
    Dict[str, float]
        Observed statistics and their permutation p-values.
    """
    pos, neg, names = extract_null_matrices(results)
    n_markers, n_perm = pos.shape

    obs_p = observed_marker_pvalues(results, names)
    perm_p = permutation_marker_pvalues(pos, neg)

    # Statistic 1 — how many markers look significant.
    t_obs = int(np.sum(obs_p <= alpha))
    t_null = np.sum(perm_p <= alpha, axis=0)
    p_count = (np.sum(t_null >= t_obs) + 1) / (n_perm + 1)

    # Statistic 2 — the single best marker (Tippett / FWER across markers).
    min_obs = float(obs_p.min())
    min_null = perm_p.min(axis=0)
    p_minp = (np.sum(min_null <= min_obs) + 1) / (n_perm + 1)

    best = names[int(np.argmin(obs_p))]

    return {
        "n_markers": n_markers,
        "n_permutations": n_perm,
        "count_observed": t_obs,
        "count_null_mean": float(t_null.mean()),
        "count_p_omnibus": float(p_count),
        "best_marker": best,
        "min_p_observed": min_obs,
        "min_p_fwer": float(p_minp),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Single model directory; default is all of them.")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    alpha = config["multiple_comparisons"]["alpha"]
    output_root = Path(config["project"]["output_path"])

    model_dirs = (
        [Path(args.model_dir)] if args.model_dir
        else sorted(d for d in output_root.iterdir() if d.is_dir())
    )

    rows = []
    for model_dir in model_dirs:
        print("=" * 80)
        print(model_dir.name)
        print("=" * 80)

        results = load_marker_results(model_dir, verbose=False)
        in_family, _ = partition_results_by_config(results, config, verbose=False)

        for marker_type in ("evoked", "state", "sleep"):
            family = [r for r in in_family if r.get("marker_type") == marker_type]
            if not family:
                continue

            stats = run_omnibus(family, alpha)
            stats["model"] = model_dir.name
            stats["target"] = model_dir.name.split("__target_")[-1]
            stats["family"] = marker_type
            rows.append(stats)

            print(
                f"  [{marker_type}] m={stats['n_markers']}  "
                f"markers p<={alpha}: {stats['count_observed']} "
                f"(null mean {stats['count_null_mean']:.2f})  "
                f"-> omnibus p = {stats['count_p_omnibus']:.5f}"
            )
            print(
                f"       best: {stats['best_marker']} "
                f"p={stats['min_p_observed']:.5f} -> FWER p = {stats['min_p_fwer']:.5f}"
            )

    df = pd.DataFrame(rows)
    cols = ["model", "target", "family", "n_markers", "n_permutations",
            "count_observed", "count_null_mean", "count_p_omnibus",
            "best_marker", "min_p_observed", "min_p_fwer"]
    df = df[cols]

    out_path = output_root / OUTPUT_NAME
    df.to_csv(out_path, index=False)
    print(f"\n✓ Omnibus results saved to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
