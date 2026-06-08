"""
Cross-decoding generalization matrix for MW probe dimensions (predictions-based).

The matrix is assembled POST-HOC from the per-probe predictions that each
per-dimension classification run already saves — nothing is retrained and no
data is reloaded. Each run writes a ``*_consolidated_sample_predictions.csv``
with, per held-out probe: ``subject, task, probe_number``, ``y_true_first`` (that
dimension's binary label) and ``proba_mean`` (the model's out-of-fold score,
averaged across runs).

Cell (model M -> dimension D) = AUC of M's scores against D's labels, over the
probes present in BOTH runs (the intersection of their native trial sets). The
diagonal (M -> M) reproduces that dimension's own AUC. Off-diagonal cells reveal
whether a model optimized for one dimension also separates another — i.e.
whether the neural signatures are shared (high) or distinct (~0.5).

Because each pair is scored on the intersection of its native trial sets, the
off-diagonal cells of a given row use different probe counts; the per-cell
overlap is reported alongside the AUC. This is a separate, specificity-focused
analysis that reuses — and never alters — the per-dimension classifications.
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Matrix assembly from saved predictions
# =============================================================================
def cross_decode_from_predictions(
    pred_tables: Dict[str, pd.DataFrame],
    proba_col: str = "proba_mean",
    label_col: str = "y_true_first",
    id_cols: Sequence[str] = ("subject", "task", "probe_number"),
    subject_col: str = "subject",
    min_overlap: int = 8,
) -> Dict[str, object]:
    """Build the cross-decoding AUC matrix from saved per-probe predictions.

    AUC is computed PER SUBJECT and then averaged across subjects, matching how
    the per-dimension pipelines report AUC (per subject / per held-out fold, then
    averaged). Pooling all subjects' probes into one AUC would mix per-subject
    score scales and distort the result; per-subject averaging also makes the
    diagonal reproduce each dimension's headline AUC.

    Parameters
    ----------
    pred_tables : Dict[str, pd.DataFrame]
        Mapping ``dimension -> consolidated predictions table``. Each table must
        contain ``id_cols``, ``subject_col``, ``proba_col`` (that dimension's
        model score per probe) and ``label_col`` (that dimension's binary label).
    proba_col : str
        Column holding the model's out-of-fold probability.
    label_col : str
        Column holding the binary label.
    id_cols : Sequence[str]
        Columns that jointly identify a probe across runs.
    subject_col : str
        Column to group by when computing AUC (the averaging unit).
    min_overlap : int
        Minimum shared probes a subject must contribute (with both classes) to be
        scored. Subjects below this are skipped; a cell with no qualifying subject
        is NaN.

    Returns
    -------
    Dict[str, object]
        ``{'mean': DataFrame, 'n_overlap': DataFrame, 'n_subjects': DataFrame,
        'dimensions': list}`` indexed train (rows) × test (cols). ``n_overlap`` is
        the total shared probes; ``n_subjects`` the subjects contributing an AUC.
    """
    from sklearn.metrics import roc_auc_score

    dimensions = list(pred_tables.keys())
    id_cols = list(id_cols)

    # Per dimension, frame indexed by probe-id with subject, score and label.
    score_frames: Dict[str, pd.DataFrame] = {}
    label_frames: Dict[str, pd.DataFrame] = {}
    for dim, table in pred_tables.items():
        missing = [c for c in id_cols + [subject_col, proba_col, label_col]
                   if c not in table.columns]
        if missing:
            raise ValueError(f"Prediction table for '{dim}' is missing columns: {missing}")
        pid = table[id_cols].astype(str).agg("_".join, axis=1)
        score_frames[dim] = pd.DataFrame(
            {"subject": table[subject_col].values, "score": table[proba_col].values}, index=pid)
        label_frames[dim] = pd.DataFrame(
            {"label": table[label_col].values}, index=pid)

    auc = pd.DataFrame(np.nan, index=dimensions, columns=dimensions, dtype=float)
    n_overlap = pd.DataFrame(0, index=dimensions, columns=dimensions, dtype=int)
    n_subjects = pd.DataFrame(0, index=dimensions, columns=dimensions, dtype=int)

    for train_dim in dimensions:
        for test_dim in dimensions:
            shared = score_frames[train_dim].index.intersection(label_frames[test_dim].index)
            n_overlap.loc[train_dim, test_dim] = len(shared)
            if len(shared) == 0:
                continue
            merged = pd.DataFrame({
                "subject": score_frames[train_dim].loc[shared, "subject"].values,
                "score": score_frames[train_dim].loc[shared, "score"].astype(float).values,
                "label": label_frames[test_dim].loc[shared, "label"].astype(int).values,
            })
            per_subject_auc = []
            for _, grp in merged.groupby("subject"):
                if len(grp) < min_overlap or grp["label"].nunique() < 2:
                    continue
                per_subject_auc.append(roc_auc_score(grp["label"].values, grp["score"].values))
            n_subjects.loc[train_dim, test_dim] = len(per_subject_auc)
            if per_subject_auc:
                auc.loc[train_dim, test_dim] = float(np.mean(per_subject_auc))

    return {"mean": auc, "n_overlap": n_overlap, "n_subjects": n_subjects,
            "dimensions": dimensions}


# =============================================================================
# Permutation null that preserves label correlation
# =============================================================================
def cross_decode_permutation_test(
    pred_tables: Dict[str, pd.DataFrame],
    n_perm: int = 500,
    random_state: int = 42,
    proba_col: str = "proba_mean",
    label_col: str = "y_true_first",
    id_cols: Sequence[str] = ("subject", "task", "probe_number"),
    subject_col: str = "subject",
    min_overlap: int = 8,
) -> Dict[str, object]:
    """Test each cell against a null that PRESERVES the label correlation.

    The off-diagonal AUC(model M -> labels of D) is inflated mechanically when
    M and D labels are correlated. To isolate D-specific neural information, the
    null shuffles M's scores WITHIN each (subject, M-label) group: this keeps the
    score<->M-label link (hence the M-D label correlation pathway) intact while
    destroying any extra ranking of D within an M-class. If the observed AUC
    exceeds this null, the model carries D information beyond the label overlap.

    The diagonal (M -> M) is non-significant by construction (within-class
    shuffling preserves the between-class ordering that drives the self-AUC).

    Parameters
    ----------
    pred_tables : Dict[str, pd.DataFrame]
        Per-dimension consolidated predictions (see
        :func:`cross_decode_from_predictions`).
    n_perm : int
        Number of within-class permutations.
    random_state : int
        Seed for the permutation RNG.
    min_overlap : int
        Minimum shared probes a subject must contribute (with both classes).

    Returns
    -------
    Dict[str, object]
        ``{'mean', 'null_mean', 'pvalue', 'n_overlap', 'dimensions'}`` indexed
        train (rows) × test (cols). ``pvalue`` is ``(1 + #{null >= obs}) / (1 + n)``.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    dimensions = list(pred_tables.keys())
    id_cols = list(id_cols)

    frames: Dict[str, pd.DataFrame] = {}
    for dim, table in pred_tables.items():
        pid = table[id_cols].astype(str).agg("_".join, axis=1)
        frames[dim] = pd.DataFrame({
            "subject": table[subject_col].values,
            "score": table[proba_col].astype(float).values,
            "label": table[label_col].astype(int).values,
        }, index=pid)

    auc = pd.DataFrame(np.nan, index=dimensions, columns=dimensions, dtype=float)
    null_mean = pd.DataFrame(np.nan, index=dimensions, columns=dimensions, dtype=float)
    pvalue = pd.DataFrame(np.nan, index=dimensions, columns=dimensions, dtype=float)
    n_overlap = pd.DataFrame(0, index=dimensions, columns=dimensions, dtype=int)

    def _avg_auc(score_arr, d_label, subject_pos):
        """Mean over qualifying subjects of AUC(d_label, score)."""
        vals = []
        for pos in subject_pos:
            vals.append(roc_auc_score(d_label[pos], score_arr[pos]))
        return float(np.mean(vals)) if vals else np.nan

    for train_dim in dimensions:
        for test_dim in dimensions:
            shared = frames[train_dim].index.intersection(frames[test_dim].index)
            n_overlap.loc[train_dim, test_dim] = len(shared)
            if len(shared) == 0:
                continue
            subj = frames[train_dim].loc[shared, "subject"].values
            score = frames[train_dim].loc[shared, "score"].to_numpy(dtype=float)
            m_label = frames[train_dim].loc[shared, "label"].to_numpy(dtype=int)
            d_label = frames[test_dim].loc[shared, "label"].to_numpy(dtype=int)

            # Qualifying subjects for scoring (>= min_overlap probes, both D-classes).
            subject_pos = []
            for s in pd.unique(subj):
                pos = np.where(subj == s)[0]
                if len(pos) >= min_overlap and len(np.unique(d_label[pos])) == 2:
                    subject_pos.append(pos)
            if not subject_pos:
                continue

            observed = _avg_auc(score, d_label, subject_pos)
            auc.loc[train_dim, test_dim] = observed
            if np.isnan(observed):
                continue

            # Groups to shuffle within: (subject, M-label).
            shuffle_pos = [np.where((subj == s) & (m_label == c))[0]
                           for s in pd.unique(subj) for c in (0, 1)]
            shuffle_pos = [p for p in shuffle_pos if len(p) > 1]

            null = np.empty(n_perm, dtype=float)
            for p in range(n_perm):
                perm = score.copy()
                for grp in shuffle_pos:
                    perm[grp] = rng.permutation(perm[grp])
                null[p] = _avg_auc(perm, d_label, subject_pos)
            null = null[~np.isnan(null)]
            null_mean.loc[train_dim, test_dim] = float(np.mean(null))
            pvalue.loc[train_dim, test_dim] = (1 + int(np.sum(null >= observed))) / (1 + len(null))

    return {"mean": auc, "null_mean": null_mean, "pvalue": pvalue,
            "n_overlap": n_overlap, "dimensions": dimensions}


# =============================================================================
# Output (CSV + heatmap)
# =============================================================================
def save_cross_decoding_outputs(
    result: Dict[str, object],
    out_dir: str,
    title: str = "Cross-decoding AUC",
) -> None:
    """Write the AUC matrix (and per-cell overlap) to CSV plus a heatmap PNG.

    Parameters
    ----------
    result : Dict[str, object]
        Output of :func:`cross_decode_from_predictions`.
    out_dir : str
        Directory to write into (created if missing).
    title : str
        Heatmap title.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    mean_df: pd.DataFrame = result["mean"]
    mean_df.to_csv(os.path.join(out_dir, "cross_decoding_mean_auc.csv"))
    n_overlap: pd.DataFrame = result.get("n_overlap")
    if n_overlap is not None:
        n_overlap.to_csv(os.path.join(out_dir, "cross_decoding_n_overlap.csv"))

    fig, ax = plt.subplots(figsize=(1.6 * len(mean_df) + 2, 1.6 * len(mean_df) + 1))
    data = mean_df.values.astype(float)
    finite = data[np.isfinite(data)]
    vmax = max(0.6, float(np.nanmax(finite))) if finite.size else 0.6
    im = ax.imshow(data, vmin=0.4, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(len(mean_df.columns)))
    ax.set_yticks(range(len(mean_df.index)))
    ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(mean_df.index)
    ax.set_xlabel("Tested dimension (labels)")
    ax.set_ylabel("Trained dimension (model)")
    ax.set_title(title)
    for i in range(len(mean_df.index)):
        for j in range(len(mean_df.columns)):
            val = data[i, j]
            if np.isnan(val):
                txt = "n/a"
            elif n_overlap is not None:
                txt = f"{val:.2f}\n(n={n_overlap.iloc[i, j]})"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_decoding_heatmap.png"), dpi=150)
    plt.close(fig)
