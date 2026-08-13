"""Recompute headline AUCs/p-values for the cross-dimension residualized contrasts.

Sibling of ``recompute_headline_numbers.py``: reuses its (not reimplemented)
statistics — ``load_group_data``, ``load_subject_data``, ``empirical_pvalue``,
``fdr_correct`` — against the residualized contrasts added in
docs/superpowers/specs/2026-07-30-cross-dimension-residualized-contrasts-design.md,
instead of the canonical 5 raw dimensions. Kept separate from the canonical
script's ``DIMENSIONS`` list so this exploratory set never gets silently
folded into the Section 3 headline numbers.
"""

# =============================================================================
# Imports
# =============================================================================
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_headline_numbers import (
    load_group_data,
    load_subject_data,
    empirical_pvalue,
    fdr_correct,
)

# =============================================================================
# Configuration
# =============================================================================
# ws_dir == loso_dir for all of these: both pipelines' configs use identical
# contrast names for the residualized set (unlike the raw dimensions, where
# WS/LOSO historically diverged in casing).
DIMENSIONS: list[dict[str, str]] = [
    {"key": "onoff_res", "ws_dir": "onoff_within_median_res", "loso_dir": "onoff_within_median_res", "label": "onoff_res"},
    {"key": "valence_res", "ws_dir": "valence_within_median_res", "loso_dir": "valence_within_median_res", "label": "valence_res"},
    {"key": "selfother_res", "ws_dir": "selfother_within_median_res", "loso_dir": "selfother_within_median_res", "label": "selfother_res"},
    {"key": "time_res", "ws_dir": "time_within_median_res", "loso_dir": "time_within_median_res", "label": "time_res"},
    {"key": "confidence_res", "ws_dir": "confidence_within_median_res", "loso_dir": "confidence_within_median_res", "label": "confidence_res"},
    {"key": "valence_sq_res_cross", "ws_dir": "valence_sq_res_cross", "loso_dir": "valence_sq_res_cross", "label": "valence_sq_res_cross"},
    {"key": "time_sq_res_cross", "ws_dir": "time_sq_res_cross", "loso_dir": "time_sq_res_cross", "label": "time_sq_res_cross"},
]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    for pipeline, label in [("ws", "WITHIN-SUBJECT"), ("loso", "LOSO")]:
        print(f"\n===== {label} - group level (residualized contrasts) =====")
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
                f"{dim_label:22s} n_true={n_t:4d} n_perm={n_p:4d} "
                f"mean_AUC={mean_t:.4f} median_AUC={med_t:.4f} "
                f"null_mean={mean_p:.4f} null_sd={std_p:.4f} "
                f"p_raw={pv:.4f} p_FDR={p_adj:.4f}"
            )

        if pipeline == "ws":
            print(f"\n===== {label} - individual (per-subject) level (residualized) =====")
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
                    f"{dim_info['label']:22s} n_subjects={len(subj_list):3d} "
                    f"n_sig(FDR<.05)={n_sig:3d}  ({n_sig}/{len(subj_list)})"
                )


if __name__ == "__main__":
    main()
