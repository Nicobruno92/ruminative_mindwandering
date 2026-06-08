#!/usr/bin/env python
"""
Permutation null for the cross-decoding matrix (label-correlation-preserving).

Tests whether each off-diagonal AUC exceeds what the M<->D label correlation
alone produces (within-(subject, M-class) shuffle of the model scores). Saves the
p-values, the null means, and a significance-annotated heatmap NEXT TO the
observed matrix, so the decisive result is on disk (not just printed).

USAGE:
    python run_permutation.py --config config.yaml --scheme WithinSubject
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import load_config, get_project_root
from utils.cross_decoding_utils import cross_decode_permutation_test
from run_cross_decoding import load_prediction_table


def main() -> None:
    """Compute and persist the label-correlation-preserving permutation null."""
    parser = argparse.ArgumentParser(description="Cross-decoding permutation null.")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--scheme", type=str, default="WithinSubject")
    parser.add_argument("--n_perm", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    cd = config["cross_decoding"]
    results_root = cd["results_root"]
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)

    family = cd.get("family", "all")
    model = cd.get("model", "rf")
    dims = {**cd["dimensions"], **cd.get("scheme_folder_overrides", {}).get(args.scheme, {})}

    pred_tables = {
        dim: load_prediction_table(
            results_root, args.scheme, folder, family, model,
            cd.get("predictions_glob", "*consolidated_sample_predictions.csv"),
            cd.get("exclude_substring", "permutation"),
        )
        for dim, folder in dims.items()
    }

    res = cross_decode_permutation_test(
        pred_tables, n_perm=args.n_perm, min_overlap=cd.get("min_overlap", 8),
        random_state=config.get("random_seed", 42),
    )

    out_dir = os.path.join(results_root, "cross_decoding", f"{args.scheme}_{model}_{family}")
    os.makedirs(out_dir, exist_ok=True)
    res["mean"].to_csv(os.path.join(out_dir, "cross_decoding_mean_auc.csv"))
    res["null_mean"].to_csv(os.path.join(out_dir, "cross_decoding_null_mean.csv"))
    res["pvalue"].to_csv(os.path.join(out_dir, "cross_decoding_pvalue.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim_list = res["dimensions"]
    m = res["mean"].values.astype(float)
    pv = res["pvalue"].values.astype(float)
    finite = m[np.isfinite(m)]
    fig, ax = plt.subplots(figsize=(1.8 * len(dim_list) + 2, 1.8 * len(dim_list) + 1))
    im = ax.imshow(m, vmin=0.4, vmax=max(0.7, float(np.nanmax(finite))), cmap="RdBu_r")
    ax.set_xticks(range(len(dim_list)))
    ax.set_yticks(range(len(dim_list)))
    ax.set_xticklabels(dim_list, rotation=45, ha="right")
    ax.set_yticklabels(dim_list)
    ax.set_xlabel("Tested dimension (labels)")
    ax.set_ylabel("Trained dimension (model)")
    ax.set_title(f"Cross-decoding AUC vs label-correlation null ({args.scheme})\n"
                 f"* = exceeds null at p<0.05 (uncorrected)")
    for i in range(len(dim_list)):
        for j in range(len(dim_list)):
            v = m[i, j]
            if np.isnan(v):
                txt = "n/a"
            else:
                star = "*" if pv[i, j] < 0.05 else ""
                txt = f"{v:.2f}\np={pv[i, j]:.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_decoding_signif_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"Saved permutation results to: {out_dir}")
    print("  cross_decoding_pvalue.csv, cross_decoding_null_mean.csv, cross_decoding_signif_heatmap.png")
    print("\np-values (obs > label-correlation null):")
    print(res["pvalue"].round(3).to_string())


if __name__ == "__main__":
    main()
