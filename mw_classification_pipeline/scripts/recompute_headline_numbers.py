"""Recompute headline classification AUCs/p-values straight from raw per-run CSVs.

Duplicates (does not reimplement independently) the statistics used by
``generate_combined_classification_figure.py`` — ``load_group_data``,
``load_subject_data``, ``empirical_pvalue``, ``fdr_correct`` — but skips every
plotting dependency (kaleido/plotly), so it runs in any env with pandas/numpy/
statsmodels. Use this whenever ``results/MW_Classification/`` changes and the
README's "Current results snapshot" section needs re-deriving; never hand-edit
that section without re-running this first.

See ``README.md`` §12 ("Current results snapshot") and §13 (why the
``*_summary_averaged.csv`` / top-level ``*_runs_summary.csv`` convenience
files are not used here) for the full context.
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Configuration
# =============================================================================
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "MW_Classification"

DIMENSIONS: list[dict[str, str]] = [
    {"key": "on_off", "ws_dir": "on_vs_off_within_median", "loso_dir": "ON_vs_OFF_within_median", "label": "On/Off-Task"},
    {"key": "valence", "ws_dir": "valence_within_median", "loso_dir": "valence_within_median", "label": "Valence"},
    {"key": "time", "ws_dir": "time_within_median", "loso_dir": "time_within_median", "label": "Time"},
    {"key": "selfother", "ws_dir": "selfother_within_median", "loso_dir": "selfother_within_median", "label": "Self/Other"},
    {"key": "confidence", "ws_dir": "confidence_within_median", "loso_dir": "confidence_within_median", "label": "Confidence"},
]

# =============================================================================
# Helper functions
# =============================================================================


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV, returning None instead of raising on empty/malformed files."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def load_group_data(pipeline: str, dim_info: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_mean_aucs, perm_mean_aucs) pooled across all raw per-run files.

    Parameters
    ----------
    pipeline : str
        Either ``"ws"`` (within-subject) or ``"loso"``.
    dim_info : dict
        One entry from `DIMENSIONS`.
    """
    if pipeline == "ws":
        base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
    else:
        base = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"

    def _collect(subdir: str) -> np.ndarray:
        vals: list[float] = []
        for csv_f in sorted(base.glob(f"{subdir}/run_*/*_summary.csv")):
            df = _safe_read_csv(csv_f)
            if df is not None and "mean_auc" in df.columns:
                vals.extend(df["mean_auc"].dropna().tolist())
        return np.array(vals)

    return _collect("true_runs"), _collect("permuted_runs")


def load_subject_data(pipeline: str, dim_info: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (true_df, perm_df) with columns [subject, auc], pooled across raw per-run files."""
    if pipeline == "ws":
        base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        true_glob, perm_glob, auc_col, agg_folds = (
            "true_runs/run_*/*_ws_subject_metrics.csv",
            "permuted_runs/run_*/*_ws_subject_metrics.csv",
            "mean_auc",
            False,
        )
    else:
        base = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
        true_glob, perm_glob, auc_col, agg_folds = (
            "true_runs/run_*/*_loso_subject_metrics.csv",
            "permuted_runs/run_*/*_loso_subject_metrics.csv",
            "auc",
            True,
        )

    def _collect(glob_pat: str) -> pd.DataFrame:
        dfs = []
        for csv_f in sorted(base.glob(glob_pat)):
            df = _safe_read_csv(csv_f)
            if df is None or "subject" not in df.columns or auc_col not in df.columns:
                continue
            df["subject"] = df["subject"].astype(str)
            if agg_folds:
                df = df.groupby("subject")[auc_col].mean().reset_index()
            df = df[["subject", auc_col]].rename(columns={auc_col: "auc"}).dropna(subset=["auc"])
            dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=["subject", "auc"])
        return pd.concat(dfs, ignore_index=True)

    return _collect(true_glob), _collect(perm_glob)


def empirical_pvalue(true_vals: np.ndarray, perm_vals: np.ndarray) -> float:
    """Same convention as the manuscript figure: fraction of null >= median(true)."""
    if len(true_vals) == 0 or len(perm_vals) == 0:
        return np.nan
    return float(np.mean(perm_vals >= np.median(true_vals)))


def fdr_correct(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, skipping NaNs."""
    arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(arr)
    adj = arr.copy()
    if valid.sum() > 0:
        _, adj[valid], _, _ = multipletests(arr[valid], method="fdr_bh")
    return adj


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    for pipeline, label in [("ws", "WITHIN-SUBJECT"), ("loso", "LOSO")]:
        print(f"\n===== {label} - group level =====")
        raw_pvals, rows = [], []
        for dim_info in DIMENSIONS:
            t, p = load_group_data(pipeline, dim_info)
            pv = empirical_pvalue(t, p)
            raw_pvals.append(pv)
            rows.append((
                dim_info["label"], len(t), len(p),
                np.mean(t) if len(t) else np.nan, np.median(t) if len(t) else np.nan,
                np.mean(p) if len(p) else np.nan, np.std(p) if len(p) else np.nan, pv,
            ))
        adj = fdr_correct(raw_pvals)
        for (dim_label, n_t, n_p, mean_t, med_t, mean_p, std_p, pv), p_adj in zip(rows, adj):
            print(
                f"{dim_label:14s} n_true={n_t:4d} n_perm={n_p:4d} "
                f"mean_AUC={mean_t:.4f} median_AUC={med_t:.4f} "
                f"null_mean={mean_p:.4f} null_sd={std_p:.4f} "
                f"p_raw={pv:.4f} p_FDR={p_adj:.4f}"
            )

        if pipeline == "ws":
            print(f"\n===== {label} - individual (per-subject) level =====")
            all_true, all_perm, all_subjects = {}, {}, set()
            for dim_info in DIMENSIONS:
                t_df, p_df = load_subject_data(pipeline, dim_info)
                all_true[dim_info["key"]] = t_df
                all_perm[dim_info["key"]] = p_df
                all_subjects.update(t_df["subject"].unique().tolist())
            sorted_subjects = sorted(all_subjects)

            for dim_info in DIMENSIONS:
                dk = dim_info["key"]
                t_df, p_df = all_true[dk], all_perm[dk]
                raw_list, subj_list = [], []
                for s in sorted_subjects:
                    tv = t_df.loc[t_df["subject"] == s, "auc"].values
                    if len(tv) == 0:
                        continue
                    pv_ = p_df.loc[p_df["subject"] == s, "auc"].values
                    raw_list.append(empirical_pvalue(tv, pv_))
                    subj_list.append(s)
                adj_s = fdr_correct(raw_list)
                n_sig = int(np.sum(adj_s < 0.05))
                print(
                    f"{dim_info['label']:14s} n_subjects={len(subj_list):3d} "
                    f"n_sig(FDR<.05)={n_sig:3d}  ({n_sig}/{len(subj_list)})"
                )


if __name__ == "__main__":
    main()
