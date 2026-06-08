#!/usr/bin/env python
"""
Merge LOSO classification results from parallel SLURM jobs.

After running run_cluster.sh (which submits 1 SLURM job per true run and 1 per
permutation), this script aggregates all per-job outputs into the same
consolidated files that a sequential run would have produced.

USAGE:
    python merge_loso_results.py --config config.yaml
    python merge_loso_results.py --config config.yaml --contrast ON_vs_OFF_within_median
    python merge_loso_results.py --config config.yaml --family all --model_type rf

All combinations found in config are merged unless overridden by CLI flags.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_utils import load_config, get_project_root


# =============================================================================
# HELPERS
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Merge per-job LOSO results into consolidated files")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--contrast", type=str, default=None)
    p.add_argument("--family", type=str, default=None)
    p.add_argument("--model_type", type=str, default=None)
    return p.parse_args()


def _results_dir(config: dict, contrast: str, family: str, model: str) -> Path:
    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    project_root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(project_root / results_root)
    return Path(results_root) / "LOSO" / contrast / family / model


def _read_per_run_summaries(model_dir: Path, model: str, n_runs: int) -> pd.DataFrame:
    """Stack one-row CSVs from true_runs/run_*/ into a single DataFrame."""
    rows = []
    for run_idx in range(n_runs):
        f = model_dir / "true_runs" / f"run_{run_idx}" / f"{model}_loso_summary.csv"
        if not f.exists():
            print(f"  WARNING: missing true run {run_idx}: {f}")
            continue
        df = pd.read_csv(f)
        df["run_idx"] = run_idx
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _read_per_perm_summaries(model_dir: Path, model: str, n_perms: int) -> pd.DataFrame:
    """Stack per-perm summary CSVs from permuted_runs/run_*/."""
    rows = []
    for perm_idx in range(n_perms):
        # Each perm job saves with filename based on total_n_permutations (n_perms)
        f = model_dir / "permuted_runs" / f"run_{perm_idx}" / f"{model}_permutation_{n_perms}perms_summary.csv"
        if not f.exists():
            print(f"  WARNING: missing perm {perm_idx}: {f}")
            continue
        df = pd.read_csv(f)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _read_subject_metrics(model_dir: Path, model: str, n_runs: int, kind: str = "true") -> pd.DataFrame:
    rows = []
    for idx in range(n_runs):
        if kind == "true":
            f = model_dir / "true_runs" / f"run_{idx}" / f"{model}_loso_loso_subject_metrics.csv"
        else:
            f = model_dir / "permuted_runs" / f"run_{idx}" / f"{model}_permutation_{n_runs}perms_loso_subject_metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["run_idx"] = idx
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _compute_pvalues(true_df: pd.DataFrame, perm_df: pd.DataFrame) -> dict:
    from scipy import stats as scipy_stats
    metrics = [
        ("AUC",               "mean_auc"),
        ("Balanced Accuracy", "mean_balanced_accuracy"),
        ("AUPRC",             "mean_auprc"),
        ("MCC",               "mean_mcc"),
    ]
    summary = {}
    for name, col in metrics:
        if col not in true_df.columns or col not in perm_df.columns:
            continue
        true_vals = pd.to_numeric(true_df[col], errors="coerce").dropna().values
        perm_vals = pd.to_numeric(perm_df[col], errors="coerce").dropna().values
        if len(true_vals) == 0 or len(perm_vals) == 0:
            continue
        true_mean = float(np.mean(true_vals))
        perm_mean = float(np.mean(perm_vals))
        perm_std  = float(np.std(perm_vals))
        # Add-one (Phipson & Smyth 2010): unbiased, never-zero permutation
        # p-value. Smallest reportable p is 1/(n_perm+1).
        empirical_p = float((1 + np.sum(perm_vals >= true_mean)) / (1 + len(perm_vals)))
        _, mwu_p = scipy_stats.mannwhitneyu(true_vals, perm_vals, alternative="greater")
        summary[col] = {
            "true_mean": true_mean,
            "perm_mean": perm_mean,
            "perm_std":  perm_std,
            "empirical_p": empirical_p,
            "mwu_p": float(mwu_p),
        }
        print(f"  {name}: null={perm_mean:.4f}±{perm_std:.4f}, "
              f"true={true_mean:.4f}, p_emp={empirical_p:.4f}, p_mwu={mwu_p:.4f}")
    return summary


# =============================================================================
# MAIN MERGE
# =============================================================================

def merge_one(config: dict, contrast: str, family: str, model: str) -> None:
    n_runs  = config.get("n_runs", 10)
    n_perms = config.get("permutation_runs", 10)
    model_dir = _results_dir(config, contrast, family, model)

    print(f"\n{'='*60}")
    print(f"Merging: {contrast} / {family} / {model}")
    print(f"  Dir: {model_dir}")
    print(f"  Expecting {n_runs} true runs, {n_perms} perms")
    print(f"{'='*60}")

    if not model_dir.exists():
        print(f"  ERROR: results directory does not exist: {model_dir}")
        return

    filename_base = f"{model}_loso_{n_runs}runs"

    # ------------------------------------------------------------------
    # True runs
    # ------------------------------------------------------------------
    true_df = _read_per_run_summaries(model_dir, model, n_runs)
    if true_df.empty:
        print("  No true-run summaries found — skipping.")
    else:
        out = model_dir / f"{filename_base}_runs_summary.csv"
        true_df.to_csv(out, index=False)
        print(f"  Saved {len(true_df)} true runs → {out.name}")

        # Per-subject metrics (averaged across runs)
        sub_all = _read_subject_metrics(model_dir, model, n_runs, kind="true")
        if not sub_all.empty:
            num_cols = [c for c in ["auc", "auprc", "mcc", "balanced_accuracy",
                                     "precision", "recall", "f1"] if c in sub_all.columns]
            sub_avg = sub_all.groupby("subject")[num_cols].mean().reset_index()
            sub_all.to_csv(model_dir / f"{filename_base}_loso_subject_metrics_all_runs.csv", index=False)
            sub_avg.to_csv(model_dir / f"{filename_base}_loso_subject_metrics.csv", index=False)
            print(f"  Saved subject metrics ({len(sub_avg)} subjects)")

            # Summary stats
            for metric in ["auc", "auprc", "mcc", "balanced_accuracy"]:
                if metric in sub_avg.columns:
                    v = sub_avg[metric]
                    print(f"    {metric.upper()}: {v.mean():.4f} ± {v.std():.4f}")

    # ------------------------------------------------------------------
    # Permutations
    # ------------------------------------------------------------------
    perm_df = _read_per_perm_summaries(model_dir, model, n_perms)
    if perm_df.empty:
        print("  No perm summaries found — skipping p-value computation.")
    else:
        perm_filename = f"{model}_permutation_{n_perms}perms_summary.csv"
        perm_df.to_csv(model_dir / perm_filename, index=False)
        print(f"  Saved {len(perm_df)} perm runs → {perm_filename}")

        if not true_df.empty:
            print("\n  P-values (empirical + Mann-Whitney U):")
            pval_summary = _compute_pvalues(true_df, perm_df)
            pval_df = pd.DataFrame(pval_summary).T.reset_index().rename(columns={"index": "metric"})
            pval_df.to_csv(model_dir / f"{filename_base}_pvalues.csv", index=False)
            print(f"  Saved p-values → {filename_base}_pvalues.csv")

    print(f"\n  Done: {model_dir}")


def main():
    args = _parse_args()
    config = load_config(args.config)

    models    = config.get("model_type", ["rf"])
    if isinstance(models, str):
        models = [models]
    contrasts = config.get("run_contrasts", list(config.get("label_contrasts", {}).keys()))
    families  = config.get("run_families", ["all"])

    if args.contrast:
        contrasts = [args.contrast]
    if args.family:
        families = [args.family]
    if args.model_type:
        models = [args.model_type]

    for model in models:
        for contrast in contrasts:
            for family in families:
                merge_one(config, contrast, family, model)

    print("\nAll merges complete.")


if __name__ == "__main__":
    main()
