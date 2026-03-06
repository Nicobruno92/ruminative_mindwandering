#!/usr/bin/env python
"""
Retroactive migration: generate per-run _loso_subject_metrics.csv files.

For each LOSO run directory that contains a *_sample_predictions.csv but
lacks the corresponding *_loso_subject_metrics.csv, this script reads the
sample-level predictions and computes per-subject AUC, AUPRC, MCC,
balanced accuracy, precision, recall, and F1.

This provides the multi-run subject-level data required for the ridgeline
distribution plots in generate_pipeline_plots.py.

USAGE (from project root):
    conda activate ML
    python mw_classification_pipeline/scripts/migrate_loso_subject_metrics.py

Or target a specific results dir:
    python mw_classification_pipeline/scripts/migrate_loso_subject_metrics.py \
        --results_root mw_classification_pipeline/results/MW_Classification/LOSO
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_RESULTS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "results", "MW_Classification", "LOSO"
)
KNOWN_MODEL_TYPES = {"lr", "rf", "xgb"}


# =============================================================================
# HELPERS
# =============================================================================

def compute_subject_metrics(sub_df: pd.DataFrame, fold_idx: int, subject) -> dict:
    """
    Compute classification metrics for a single LOSO fold (subject).

    Parameters
    ----------
    sub_df : pd.DataFrame
        All rows from *_sample_predictions.csv belonging to this subject.
    fold_idx : int
        Index of this fold in the LOSO run.
    subject
        Subject identifier.

    Returns
    -------
    dict
        Per-subject metric dictionary.
    """
    y_true = sub_df["y_true"].values
    y_pred = sub_df["y_pred"].values

    n_unique = len(np.unique(y_true))
    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_samples - n_positive

    # AUC / AUPRC require both classes to be present
    if n_unique > 1 and "y_proba" in sub_df.columns:
        y_proba = sub_df["y_proba"].values
        auc = float(roc_auc_score(y_true, y_proba))
        auprc = float(average_precision_score(y_true, y_proba))
    else:
        auc = 0.5
        auprc = float(n_positive / n_samples) if n_samples > 0 else 0.0

    mcc = float(matthews_corrcoef(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "subject": subject,
        "fold_idx": fold_idx,
        "auc": auc,
        "auprc": auprc,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _find_sample_predictions(run_dir: str) -> Optional[str]:
    """Return the path to the *_sample_predictions.csv in a run dir, or None."""
    for fname in os.listdir(run_dir):
        if fname.endswith("_sample_predictions.csv"):
            return os.path.join(run_dir, fname)
    return None


def _target_metrics_path(run_dir: str, sample_csv_path: str) -> str:
    """Derive the target _loso_subject_metrics.csv path from the sample CSV name."""
    base = Path(sample_csv_path).stem  # e.g. lr_loso_20runs_sample_predictions
    metrics_base = base.replace("_sample_predictions", "_loso_subject_metrics")
    return os.path.join(run_dir, f"{metrics_base}.csv")


def process_run_dir(run_dir: str) -> bool:
    """
    Compute and save *_loso_subject_metrics.csv for one run directory.

    Parameters
    ----------
    run_dir : str
        Path to a single run* directory.

    Returns
    -------
    bool
        True if a new file was created, False if skipped.
    """
    sample_csv = _find_sample_predictions(run_dir)
    if sample_csv is None:
        return False

    target_path = _target_metrics_path(run_dir, sample_csv)
    if os.path.exists(target_path):
        return False  # Already present

    preds = pd.read_csv(sample_csv)
    assert "subject" in preds.columns, f"No 'subject' column in {sample_csv}"
    assert "y_true" in preds.columns, f"No 'y_true' column in {sample_csv}"
    assert "y_pred" in preds.columns, f"No 'y_pred' column in {sample_csv}"

    subject_metrics = []
    for fold_idx, (subject, sub_df) in enumerate(preds.groupby("subject", sort=True)):
        subject_metrics.append(
            compute_subject_metrics(sub_df, fold_idx, subject)
        )

    pd.DataFrame(subject_metrics).to_csv(target_path, index=False)
    return True


def scan_and_migrate(results_root: str) -> None:
    """
    Walk *results_root* and migrate all run directories.

    Parameters
    ----------
    results_root : str
        Root under which to recursively find ``runs/run*`` directories.
    """
    results_root = os.path.abspath(results_root)
    assert os.path.isdir(results_root), f"results_root not found: {results_root}"

    created = 0
    skipped = 0

    for dirpath, dirnames, _ in os.walk(results_root):
        # Only process dirs named run* that live inside a runs/ parent
        parent = Path(dirpath).name
        if parent != "runs":
            continue

        # Also check that the grandparent model dir is a known model type
        model_dir = Path(dirpath).parent
        if model_dir.name not in KNOWN_MODEL_TYPES | {"permutations"}:
            # Permutation run dirs are also valid
            pass

        for run_folder in sorted(dirnames):
            if not run_folder.startswith("run"):
                continue
            run_path = os.path.join(dirpath, run_folder)
            # Skip WithinSubject dirs – they use _ws_subject_metrics.csv already
            if "WithinSubject" in run_path:
                skipped += 1
                continue
            success = process_run_dir(run_path)
            if success:
                created += 1
                print(f"  Created: {run_path}")
            else:
                skipped += 1

    print(f"\nMigration complete: {created} created, {skipped} skipped.")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root results directory to scan recursively. "
            f"Defaults to: {DEFAULT_RESULTS_ROOT}"
        ),
    )
    args = parser.parse_args()
    scan_and_migrate(args.results_root)


if __name__ == "__main__":
    main()
