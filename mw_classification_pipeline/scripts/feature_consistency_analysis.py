#!/usr/bin/env python
"""
WS <-> LOSO feature consistency, measured per subject.

The question
-----------
Does the model trained on a subject's *own* data rely on the same EEG features
as the group model trained on everyone *except* that subject? A within-subject
AUC well above the LOSO AUC says the two disagree somewhere; this script asks
where, and how much.

Why per subject
---------------
The previous analysis correlated a cross-subject *average* of within-subject
importances against a single group-level LOSO profile. Three things break that:

1. Averaging 29 heterogeneous per-subject profiles produces a near-uniform
   vector that represents no model anyone fitted. Under the mRMR configuration
   in force when those figures were made, the LOSO side stayed sparse, so the
   apparent "LOSO concentrates, WS is diffuse" contrast was arithmetic rather
   than a finding.
2. It answers a question nobody asked ("does the average subject look like the
   group?") instead of the one that matters ("does *this* subject look like the
   group's view of them?").
3. Its correlation was annotated over a top-N-of-WS union top-N-of-LOSO subset,
   which manufactures a steep negative slope out of noise by pure selection
   geometry (measured: r = 0.04 over all features vs r = -0.86 over the union,
   the latter sitting at the 25th percentile of a null that destroys the pairing
   entirely).

Here each subject S contributes one paired comparison, both sides built from
mean(|SHAP|) over *the same trials* of S, at the same 175-feature resolution:

    WS(S)   — S's own model's SHAP on S's trials
    LOSO(S) — the model trained WITHOUT S, its SHAP on those same trials

LOSO never saw S, so the comparison is not circular.

What is reported
----------------
Two questions that a single agreement number would conflate — a marker can rank
identically in both pipelines and still mean the opposite thing:

    rho_magnitude     Spearman of mean(|SHAP|) profiles     "same markers?"
    sign_concordance  fraction of subjects whose own model's direction for a
                      marker matches that of the model trained without them,
                      against an exact binomial null at 0.5   "same direction?"

Direction is the correlation between a feature's SHAP value and its own value
across trials: positive means higher values of the feature push the model toward
the positive class. A marker's nine ROI columns are combined as a mean weighted
by each column's mean(|SHAP|), so an ROI the model barely uses cannot outvote
the ones carrying it.

Every correlation is reported against its **noise ceiling** — the split-half
reliability of each side, Spearman-Brown corrected. A bare rho = 0.2 is
uninterpretable: it may be a third of what is achievable or all of it.

A companion *selection-overlap* statistic (Jaccard of the chosen feature sets
against a hypergeometric null) was designed for this analysis and is
deliberately **not** computed. Both pipelines now run with
``feature_selection.method: "none"``, so the random forest is fitted on all 175
columns and every feature is used: the selected sets are both the full set, the
Jaccard is identically 1, and the LOSO selection-frequency profile is constant,
which makes a rank correlation undefined. Reporting J = 1 as agreement would be
reporting the config back as a result. Only the number of features each side
actually assigns non-zero SHAP to is carried through, descriptively.

USAGE
-----
    conda activate ML
    python mw_classification_pipeline/scripts/feature_consistency_analysis.py \
        --config mw_classification_pipeline/scripts/config_feature_consistency.yaml
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, rankdata
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict:
    """Read the YAML configuration file."""
    return yaml.safe_load(config_path.read_text())


# =============================================================================
# Correlation helpers
# =============================================================================


def _pearson_rows(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Pearson correlation of a single vector against every row of a matrix.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    Y : np.ndarray, shape (m, n)

    Returns
    -------
    np.ndarray, shape (m,)
        Correlation of ``x`` with each row of ``Y``.
    """
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(xc) * np.linalg.norm(Yc, axis=1)
    return (Yc @ xc) / denom


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation, with average ranks for ties.

    Ties matter here: profiles can carry blocks of exact zeros (features a
    forest never split on), and average ranks are what keeps those blocks from
    dominating.
    """
    xr, yr = rankdata(x), rankdata(y)
    if xr.std() == 0 or yr.std() == 0:
        raise ValueError(
            "Constant profile: every feature has identical importance, so a rank "
            "correlation is undefined. This means a model produced no SHAP "
            "variation at all — investigate the upstream run, do not skip it."
        )
    return float(np.corrcoef(xr, yr)[0, 1])


def spearman_permutation_test(
    x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """
    Spearman correlation with a pairing-permutation p-value.

    The null shuffles which WS feature is matched to which LOSO feature. It
    preserves each side's marginal distribution *and* its tie structure, which
    the asymptotic Spearman p-value does not — and both profiles are heavily
    tied at zero. The test is one-sided: only positive correlation is evidence
    of consistency, so a negative rho is not "consistency in the other
    direction".

    Returns
    -------
    tuple[float, float]
        (rho, p_value). The p-value uses the +1 correction, so it can never be
        exactly 0 and is bounded below by 1 / (n_permutations + 1).
    """
    xr, yr = rankdata(x), rankdata(y)
    observed = float(np.corrcoef(xr, yr)[0, 1])
    shuffled = rng.permuted(np.tile(yr, (n_permutations, 1)), axis=1)
    null = _pearson_rows(xr, shuffled)
    p = (1.0 + np.sum(null >= observed)) / (1.0 + n_permutations)
    return observed, float(p)


def spearman_brown(half_reliability: float) -> float:
    """
    Extrapolate a split-half correlation to full test length.

    Parameters
    ----------
    half_reliability : float
        Correlation between two profiles each built from half the runs.

    Returns
    -------
    float
        Estimated reliability of a profile built from all runs. Negative input
        yields a value that is not a valid reliability; it is returned as-is so
        the caller sees it rather than a silently clamped number.
    """
    return 2.0 * half_reliability / (1.0 + half_reliability)


# =============================================================================
# Data loading
# =============================================================================


def _run_index(run_dir: Path) -> int:
    """Extract the integer run index from a ``run_N`` directory name."""
    match = re.fullmatch(r"run_(\d+)", run_dir.name)
    if match is None:
        raise ValueError(f"Unexpected run directory name: {run_dir}")
    return int(match.group(1))


def _columnwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pearson correlation of matching columns of two equally shaped matrices.

    Returns ``np.nan`` for any column where either side has zero variance —
    a feature that is constant across a subject's trials, or one the forest
    never split on, has no defined direction. That is reported as missing
    rather than as a direction of zero, which would be a measured null.
    """
    a_centred = a - a.mean(axis=0)
    b_centred = b - b.mean(axis=0)
    numerator = (a_centred * b_centred).sum(axis=0)
    denominator = np.sqrt((a_centred ** 2).sum(axis=0) * (b_centred ** 2).sum(axis=0))
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)


def load_pipeline_profiles(
    base_dir: Path, shap_glob: str, pred_glob: str, expected_n_features: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    Build per-subject SHAP profiles for one pipeline, one run at a time.

    Both pipelines store, per run, a stacked SHAP matrix over all trials of all
    subjects plus a sample-predictions table with one row per trial in the same
    order. Segmenting by the table's ``subject`` column recovers each subject's
    own trials; for the within-subject pipeline this reproduces the per-subject
    pickles exactly (verified), while being ~29x fewer file reads.

    Returns
    -------
    tuple
        ``(magnitude, selection, direction, feature_names)``. Each dict maps a
        subject to an array of shape ``(n_runs, n_features)``:
        ``magnitude`` holds mean(|SHAP|) over that subject's trials,
        ``selection`` the boolean "this feature was used at all", and
        ``direction`` the Pearson correlation between a feature's SHAP value and
        its own value across those trials — positive meaning higher values of
        the feature push the model toward the positive class.

    Notes
    -----
    The direction is computed over **all** of a subject's trials. An earlier
    per-feature version of this statistic dropped rows with SHAP exactly 0,
    on the grounds that they were structural zeros left by mRMR's selection.
    Both pipelines now run with `feature_selection.method: "none"`, so nothing
    is unselected and those zeros are ordinary measurements — trials where the
    tree path happened not to split on that feature. Dropping them would keep
    only the trials where the feature mattered and bias every correlation.
    """
    run_dirs = sorted(base_dir.glob("true_runs/run_*"), key=_run_index)
    if not run_dirs:
        raise FileNotFoundError(f"No true_runs/run_* directories under {base_dir}")

    magnitude: dict[str, list[np.ndarray]] = {}
    selection: dict[str, list[np.ndarray]] = {}
    direction: dict[str, list[np.ndarray]] = {}
    feature_names: list[str] = []

    for run_dir in run_dirs:
        shap_paths = sorted(run_dir.glob(shap_glob))
        pred_paths = sorted(run_dir.glob(pred_glob))
        if len(shap_paths) != 1 or len(pred_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one SHAP pickle ('{shap_glob}') and one "
                f"predictions CSV ('{pred_glob}') in {run_dir}; found "
                f"{len(shap_paths)} and {len(pred_paths)}."
            )

        with open(shap_paths[0], "rb") as handle:
            payload = pickle.load(handle)
        shap_values = np.asarray(payload["shap_values"])
        feature_values = np.asarray(payload["x_test"])
        run_features = [str(f) for f in payload["feature_names"]]
        predictions = pd.read_csv(pred_paths[0])

        if feature_values.shape != shap_values.shape:
            raise ValueError(
                f"{run_dir}: x_test is {feature_values.shape} but the SHAP matrix "
                f"is {shap_values.shape}. The direction statistic pairs them "
                f"element-wise, so they must match exactly."
            )

        if shap_values.shape[0] != len(predictions):
            raise ValueError(
                f"{run_dir}: SHAP matrix has {shap_values.shape[0]} rows but the "
                f"predictions table has {len(predictions)}. They must be "
                f"row-aligned for the per-subject segmentation to be valid."
            )
        if shap_values.shape[1] != expected_n_features:
            raise ValueError(
                f"{run_dir}: {shap_values.shape[1]} features, expected "
                f"{expected_n_features}. A mismatch here means the run used a "
                f"different feature space (see the 2026-07-28 cache bug)."
            )
        if not feature_names:
            feature_names = run_features
        elif run_features != feature_names:
            raise ValueError(f"{run_dir}: feature name/order differs from run 0.")

        subjects = predictions["subject"].astype(str).str.zfill(2).to_numpy()
        for subject in np.unique(subjects):
            mask = subjects == subject
            rows = shap_values[mask]
            magnitude.setdefault(subject, []).append(np.abs(rows).mean(axis=0))
            selection.setdefault(subject, []).append(np.any(rows != 0.0, axis=0))
            direction.setdefault(subject, []).append(
                _columnwise_correlation(rows, feature_values[mask])
            )

    return (
        {s: np.vstack(v) for s, v in magnitude.items()},
        {s: np.vstack(v) for s, v in selection.items()},
        {s: np.vstack(v) for s, v in direction.items()},
        feature_names,
    )


# =============================================================================
# Marker mapping
# =============================================================================


def build_marker_map(
    feature_names: list[str], marker_regex: str, expected_n_markers: int
) -> tuple[list[str], np.ndarray]:
    """
    Collapse the feature columns onto the 23 markers of the CBPT analysis.

    Every column is ``<marker>_mean_<roi>_trimmean``; stripping the ROI segment
    recovers the marker. The hierarchy comes from the pre-existing
    cluster-permutation analysis, not from inspecting these results, so it is
    not a forking path.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Sorted marker names, and an integer array assigning each feature to its
        marker's position in that list.
    """
    pattern = re.compile(marker_regex)
    markers = []
    for name in feature_names:
        stripped = pattern.sub("", name)
        if stripped == name:
            raise ValueError(
                f"Feature '{name}' does not match the marker grammar "
                f"'{marker_regex}'. The naming convention changed; update the "
                f"config rather than letting this column fall through."
            )
        markers.append(stripped)

    unique_markers = sorted(set(markers))
    if len(unique_markers) != expected_n_markers:
        raise ValueError(
            f"Derived {len(unique_markers)} markers, expected "
            f"{expected_n_markers}: {unique_markers}"
        )
    index = {m: i for i, m in enumerate(unique_markers)}
    return unique_markers, np.array([index[m] for m in markers])


def mean_correlation(values: np.ndarray) -> float:
    """
    Average correlations in Fisher-z space, ignoring NaN.

    Parameters
    ----------
    values : np.ndarray, shape (n,)

    Returns
    -------
    float
        ``np.nan`` if nothing is finite. Correlations are compressed near
        +/-1, so a plain arithmetic mean systematically under-weights the
        strongest ones; the z transform is the standard correction and is what
        this analysis already uses to pool across runs.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    clipped = np.clip(finite, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def average_runs_fisher(directions: np.ndarray) -> np.ndarray:
    """
    Average per-run correlations across runs in Fisher-z space.

    Parameters
    ----------
    directions : np.ndarray, shape (n_runs, n_features)
        Per-run correlations, possibly containing ``np.nan``.

    Returns
    -------
    np.ndarray, shape (n_features,)
        The back-transformed mean. Averaging r directly under-weights strong
        correlations, because r is compressed near +/-1; the z transform is
        the standard correction. Values are clipped just short of +/-1 so a
        perfect run-level correlation gives a large z rather than an infinite
        one that would poison the whole average.
    """
    clipped = np.clip(directions, -0.999999, 0.999999)
    finite = np.isfinite(clipped)
    # Averaged over the finite entries explicitly rather than with nanmean: a
    # feature that is constant for a subject is NaN in every run, and nanmean
    # would emit a "mean of empty slice" warning for each such column. The
    # all-NaN case is a defined outcome here (no direction exists), not an
    # anomaly worth a warning, so it is handled rather than reported.
    z_values = np.where(finite, np.arctanh(np.where(finite, clipped, 0.0)), 0.0)
    n_finite = finite.sum(axis=0)
    z_mean = z_values.sum(axis=0) / np.where(n_finite > 0, n_finite, 1)
    return np.where(n_finite > 0, np.tanh(z_mean), np.nan)


def aggregate_direction_to_markers(
    direction: np.ndarray, weights: np.ndarray, marker_index: np.ndarray, n_markers: int
) -> np.ndarray:
    """
    Reduce per-feature directions to one direction per marker.

    Parameters
    ----------
    direction : np.ndarray, shape (n_features,)
        Per-feature SHAP-vs-value correlation.
    weights : np.ndarray, shape (n_features,)
        Per-feature mean(|SHAP|), used to weight each ROI's contribution.

    Returns
    -------
    np.ndarray, shape (n_markers,)
        Weighted mean direction per marker, ``np.nan`` where a marker has no
        usable ROI.

    Notes
    -----
    A marker spans up to nine ROI columns, each with its own direction. They
    are combined as a mean weighted by how much each ROI actually contributes
    to the model's output, so a strongly-signed direction on an ROI the model
    barely uses cannot outvote the ROIs carrying the marker. An unweighted mean
    would let near-unused columns, whose correlations are also the noisiest,
    dominate.
    """
    usable = np.isfinite(direction) & np.isfinite(weights) & (weights > 0)
    weighted_sum = np.zeros(n_markers)
    weight_total = np.zeros(n_markers)
    np.add.at(weighted_sum, marker_index[usable], (direction * weights)[usable])
    np.add.at(weight_total, marker_index[usable], weights[usable])
    return np.where(weight_total > 0, weighted_sum / np.where(weight_total > 0, weight_total, 1.0), np.nan)


def aggregate_to_markers(
    profile: np.ndarray, marker_index: np.ndarray, n_markers: int, method: str = "mean"
) -> np.ndarray:
    """
    Reduce a feature-level importance profile to one value per marker.

    Parameters
    ----------
    method : {"mean", "sum"}
        ``"mean"`` divides each marker's total by its ROI count; ``"sum"``
        leaves the raw total.

    Returns
    -------
    np.ndarray, shape (n_markers,)

    Notes
    -----
    The choice matters because the ROI counts are wildly unequal: the four
    evoked markers (P1, N1, P3a, P3b) occupy one column each while the nineteen
    sleep markers occupy nine. Summing therefore hands the sleep markers up to
    nine times the mass before any data is consulted, burying an evoked marker
    even when it is individually strong. That is an artefact of how the feature
    space was laid out, not a fact about the brain.

    ``"sum"`` answers "how much of the model's total attribution does this
    marker absorb?", which is the honest quantity if that is the question. It is
    the wrong one for ranking markers against each other — which is what every
    figure built on this does — so attribution per ROI is the default.
    """
    totals = np.zeros(n_markers, dtype=float)
    np.add.at(totals, marker_index, profile)
    if method == "sum":
        return totals
    if method != "mean":
        raise ValueError(
            f"Unknown marker aggregation '{method}'; expected 'mean' or 'sum'."
        )
    return totals / np.bincount(marker_index, minlength=n_markers).astype(float)


# =============================================================================
# Per-subject analysis
# =============================================================================


def analyse_subject(
    ws_magnitude: np.ndarray,
    lo_magnitude: np.ndarray,
    ws_selection: np.ndarray,
    lo_selection: np.ndarray,
    n_features: int,
    n_permutations: int,
    restrict_to_nonzero_union: bool,
    rng: np.random.Generator,
) -> dict:
    """
    Compute every consistency statistic for one subject.

    Parameters
    ----------
    ws_magnitude, lo_magnitude : np.ndarray, shape (n_runs, n_features)
        Per-run mean(|SHAP|) profiles.
    ws_selection, lo_selection : np.ndarray, shape (n_runs, n_features)
        Per-run boolean "this feature was used" masks.

    Returns
    -------
    dict
        The magnitude correlation with its permutation p-value, the split-half
        reliabilities, the ceiling-corrected correlation, and how many features
        each side actually used.
    """
    ws_mean, lo_mean = ws_magnitude.mean(axis=0), lo_magnitude.mean(axis=0)

    # Features neither pipeline ever used carry no information, and a shared
    # block of tied zeros would inflate every correlation mechanically. With
    # feature selection disabled this keeps nearly all 175; it still matters for
    # the within-subject side, whose per-subject forests are fitted on ~32
    # samples and may never split on some columns.
    if restrict_to_nonzero_union:
        keep = (ws_mean > 0) | (lo_mean > 0)
    else:
        keep = np.ones(n_features, dtype=bool)

    rho_mag, p_mag = spearman_permutation_test(
        ws_mean[keep], lo_mean[keep], n_permutations, rng
    )

    # Noise ceiling: odd/even split of the runs, correlated within pipeline,
    # over the SAME feature subset as the observed correlation so the two live
    # on one scale.
    odd, even = np.arange(len(ws_magnitude)) % 2 == 1, np.arange(len(ws_magnitude)) % 2 == 0
    rel_ws = spearman_brown(
        spearman(ws_magnitude[odd].mean(axis=0)[keep], ws_magnitude[even].mean(axis=0)[keep])
    )
    rel_lo = spearman_brown(
        spearman(lo_magnitude[odd].mean(axis=0)[keep], lo_magnitude[even].mean(axis=0)[keep])
    )
    ceiling = np.sqrt(rel_ws * rel_lo) if rel_ws > 0 and rel_lo > 0 else np.nan

    return {
        "n_features_compared": int(keep.sum()),
        "rho_magnitude": rho_mag,
        "p_magnitude": p_mag,
        "reliability_ws": rel_ws,
        "reliability_loso": rel_lo,
        "noise_ceiling": ceiling,
        "rho_magnitude_corrected": rho_mag / ceiling if ceiling and ceiling > 0 else np.nan,
        "pct_of_ceiling": 100.0 * rho_mag / ceiling if ceiling and ceiling > 0 else np.nan,
        # Descriptive only: with selection disabled these are near-saturated by
        # construction and carry no inference (see the module docstring).
        "n_features_used_ws": float(ws_selection.sum(axis=1).mean()),
        "n_features_used_loso": float(lo_selection.sum(axis=1).mean()),
    }


# =============================================================================
# Contrast-level analysis
# =============================================================================


def analyse_contrast(
    contrast: dict, config: dict, results_root: Path, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full analysis for one contrast.

    Returns
    -------
    tuple
        ``(per_subject, per_marker, per_subject_direction, per_feature,
        per_subject_feature_direction, group_summary)``.
    """
    stats_cfg = config["statistics"]
    aggregation = config["marker_aggregation"]
    n_features = config["expected_n_features"]
    model, family = config["model"], config["family"]

    ws_base = results_root / "WithinSubject" / contrast["ws_dir"] / family / model
    lo_base = results_root / "LOSO" / contrast["loso_dir"] / family / model

    print(f"  loading within-subject SHAP from {ws_base.relative_to(results_root)}", flush=True)
    ws_mag, ws_sel, ws_dir, ws_features = load_pipeline_profiles(
        ws_base, "*_shap_values_stacked.pkl", "*_sample_predictions.csv", n_features
    )
    print(f"  loading LOSO SHAP from {lo_base.relative_to(results_root)}", flush=True)
    lo_mag, lo_sel, lo_dir, lo_features = load_pipeline_profiles(
        lo_base, "rf_loso_*_shap_values.pkl", "*_sample_predictions.csv", n_features
    )

    if ws_features != lo_features:
        raise ValueError(
            f"{contrast['key']}: the two pipelines report different feature "
            f"names/order. Every correlation below joins by position, so this "
            f"must be fixed upstream, not worked around."
        )

    markers, marker_index = build_marker_map(
        ws_features, config["marker_regex"], config["expected_n_markers"]
    )

    subjects = sorted(set(ws_mag) & set(lo_mag))
    dropped = sorted((set(ws_mag) | set(lo_mag)) - set(subjects))
    if dropped:
        print(f"  ! subjects present in only one pipeline, excluded: {dropped}")

    # ---- per subject -------------------------------------------------------
    rows = []
    ws_marker_by_subject, lo_marker_by_subject = [], []
    ws_dir_by_subject, lo_dir_by_subject = [], []
    for subject in subjects:
        result = analyse_subject(
            ws_mag[subject], lo_mag[subject], ws_sel[subject], lo_sel[subject],
            n_features, stats_cfg["n_permutations"],
            stats_cfg["restrict_to_nonzero_union"], rng,
        )
        rows.append({"contrast": contrast["key"], "label": contrast["label"],
                     "subject": subject, **result})

        ws_weights = ws_mag[subject].mean(axis=0)
        lo_weights = lo_mag[subject].mean(axis=0)
        ws_marker_by_subject.append(
            aggregate_to_markers(ws_weights, marker_index, len(markers), aggregation)
        )
        lo_marker_by_subject.append(
            aggregate_to_markers(lo_weights, marker_index, len(markers), aggregation)
        )
        ws_dir_by_subject.append(aggregate_direction_to_markers(
            average_runs_fisher(ws_dir[subject]), ws_weights, marker_index, len(markers)
        ))
        lo_dir_by_subject.append(aggregate_direction_to_markers(
            average_runs_fisher(lo_dir[subject]), lo_weights, marker_index, len(markers)
        ))

    per_subject = pd.DataFrame(rows)
    for stat in ("magnitude",):
        _, adjusted, _, _ = multipletests(
            per_subject[f"p_{stat}"].to_numpy(), method=stats_cfg["multiple_comparisons"]
        )
        per_subject[f"p_{stat}_fdr"] = adjusted

    # ---- marker level ------------------------------------------------------
    # A different question from the per-subject profile correlation: for each
    # marker, do subjects who rely on it heavily in their own model also get it
    # weighted heavily by the group model's view of them? The unit is the
    # subject, so this measures whether *between-subject variation* in reliance
    # on that marker is shared.
    ws_marker = np.vstack(ws_marker_by_subject)
    lo_marker = np.vstack(lo_marker_by_subject)
    ws_direction = np.vstack(ws_dir_by_subject)
    lo_direction = np.vstack(lo_dir_by_subject)

    marker_rows = []
    for i, marker in enumerate(markers):
        rho, p = spearman_permutation_test(
            ws_marker[:, i], lo_marker[:, i], stats_cfg["n_permutations"], rng
        )

        # Sign concordance: the directional counterpart of the share agreement
        # above. Share asks whether the two pipelines lean on the same markers;
        # this asks whether they lean on them the same *way*. A marker can be
        # top-ranked in both and still mean opposite things per subject, which
        # is exactly the failure mode a group-level mean direction hides.
        ws_sign = ws_direction[:, i]
        lo_sign = lo_direction[:, i]
        comparable = np.isfinite(ws_sign) & np.isfinite(lo_sign)
        n_comparable = int(comparable.sum())
        n_agree = int(np.sum(np.sign(ws_sign[comparable]) == np.sign(lo_sign[comparable])))
        # Two-sided exact binomial against chance: with no shared direction a
        # subject's sign matches the group's half the time.
        p_sign = float(binomtest(n_agree, n_comparable, 0.5).pvalue) if n_comparable else np.nan

        marker_rows.append({
            "contrast": contrast["key"], "label": contrast["label"], "marker": marker,
            "rho_across_subjects": rho, "p_across_subjects": p,
            "mean_share_ws": float(ws_marker[:, i].mean() / ws_marker.sum(axis=1).mean()),
            "mean_share_loso": float(lo_marker[:, i].mean() / lo_marker.sum(axis=1).mean()),
            # Fisher-z, matching how directions are averaged across runs. A
            # plain mean of correlations under-weights the strong ones, and
            # using one convention within runs and another across subjects was
            # simply inconsistent. The correction is small here (median 0.004,
            # max 0.034 over the 161 marker x dimension cells, one sign change)
            # and moves no conclusion, but the number goes in a paper.
            "mean_direction_ws": mean_correlation(ws_sign),
            "mean_direction_loso": mean_correlation(lo_sign),
            "sign_concordance": n_agree / n_comparable if n_comparable else np.nan,
            "p_sign_concordance": p_sign,
            "n_rois": int(np.sum(marker_index == i)),
            "n_comparable_subjects": n_comparable,
            "n_subjects": len(subjects),
        })
    per_marker = pd.DataFrame(marker_rows)
    for column in ("p_across_subjects", "p_sign_concordance"):
        valid = per_marker[column].notna().to_numpy()
        adjusted_column = np.full(len(per_marker), np.nan)
        if valid.any():
            _, adjusted_column[valid], _, _ = multipletests(
                per_marker.loc[valid, column].to_numpy(),
                method=stats_cfg["multiple_comparisons"],
            )
        per_marker[f"{column}_fdr"] = adjusted_column

    # Per-subject marker directions, kept long so the figure can draw the strip
    # of individual subjects behind each marker's group value.
    direction_rows = [
        {"contrast": contrast["key"], "label": contrast["label"],
         "subject": subject, "marker": marker,
         "direction_ws": float(ws_direction[s_idx, m_idx]),
         "direction_loso": float(lo_direction[s_idx, m_idx])}
        for s_idx, subject in enumerate(subjects)
        for m_idx, marker in enumerate(markers)
    ]
    per_subject_direction = pd.DataFrame(direction_rows)

    # ---- feature level -----------------------------------------------------
    # The same directional question one resolution down, at the raw ROI column.
    # Kept alongside the marker level rather than replacing it, because the two
    # answer different questions and only one of them is well posed for a signed
    # quantity: |SHAP| is positive and additive, so averaging a marker's ROIs is
    # sound and the marker-level *share* is solid, but directions of opposite
    # sign cancel. Measured on on/off-task, ROI sign coherence is 0.73 within
    # subject and 0.77 for LOSO — nearly equal, so the ROI disagreement is real
    # rather than within-subject noise, and "the direction of PSD alpha" is not
    # a well-defined quantity when its nine ROIs point different ways.
    ws_feature_direction = np.vstack(
        [average_runs_fisher(ws_dir[s]) for s in subjects]
    )
    lo_feature_direction = np.vstack(
        [average_runs_fisher(lo_dir[s]) for s in subjects]
    )
    ws_feature_weight = np.vstack([ws_mag[s].mean(axis=0) for s in subjects])
    lo_feature_weight = np.vstack([lo_mag[s].mean(axis=0) for s in subjects])

    feature_rows = []
    for j, feature in enumerate(ws_features):
        ws_col, lo_col = ws_feature_direction[:, j], lo_feature_direction[:, j]
        comparable = np.isfinite(ws_col) & np.isfinite(lo_col)
        n_comparable = int(comparable.sum())
        n_agree = int(np.sum(np.sign(ws_col[comparable]) == np.sign(lo_col[comparable])))
        feature_rows.append({
            "contrast": contrast["key"], "label": contrast["label"],
            "feature": feature, "marker": markers[marker_index[j]],
            "mean_share_ws": float(ws_feature_weight[:, j].mean()
                                   / ws_feature_weight.sum(axis=1).mean()),
            "mean_share_loso": float(lo_feature_weight[:, j].mean()
                                     / lo_feature_weight.sum(axis=1).mean()),
            "mean_direction_ws": mean_correlation(ws_col),
            "mean_direction_loso": mean_correlation(lo_col),
            "sign_concordance": n_agree / n_comparable if n_comparable else np.nan,
            "p_sign_concordance": (float(binomtest(n_agree, n_comparable, 0.5).pvalue)
                                   if n_comparable else np.nan),
            "n_comparable_subjects": n_comparable,
        })
    per_feature = pd.DataFrame(feature_rows)
    valid = per_feature["p_sign_concordance"].notna().to_numpy()
    adjusted_column = np.full(len(per_feature), np.nan)
    if valid.any():
        _, adjusted_column[valid], _, _ = multipletests(
            per_feature.loc[valid, "p_sign_concordance"].to_numpy(),
            method=stats_cfg["multiple_comparisons"],
        )
    per_feature["p_sign_concordance_fdr"] = adjusted_column

    per_subject_feature_direction = pd.DataFrame([
        {"contrast": contrast["key"], "subject": subject, "feature": feature,
         "direction_ws": float(ws_feature_direction[s_idx, j]),
         "direction_loso": float(lo_feature_direction[s_idx, j])}
        for s_idx, subject in enumerate(subjects)
        for j, feature in enumerate(ws_features)
    ])

    # ---- legacy group-level contrast --------------------------------------
    # What the replaced figure computed: cross-subject mean profiles, one
    # correlation. Kept so the old number can be quoted as a fraction of its own
    # ceiling instead of in a vacuum.
    ws_group = np.vstack([ws_mag[s].mean(axis=0) for s in subjects]).mean(axis=0)
    lo_group = np.vstack([lo_mag[s].mean(axis=0) for s in subjects]).mean(axis=0)
    rho_group, p_group = spearman_permutation_test(
        ws_group, lo_group, stats_cfg["n_permutations"], rng
    )

    # The same group-level comparison at marker resolution. Collinear ROI
    # columns of one marker are near-substitutable for a random forest, so two
    # models can use the same underlying signal and split on different columns;
    # collapsing to the 23 markers removes that source of disagreement, and the
    # gap between this and the feature-level number measures how much of the
    # disagreement was ROI-level arbitrariness.
    rho_marker, p_marker = spearman_permutation_test(
        aggregate_to_markers(ws_group, marker_index, len(markers), aggregation),
        aggregate_to_markers(lo_group, marker_index, len(markers), aggregation),
        stats_cfg["n_permutations"], rng,
    )

    summary = {
        "contrast": contrast["key"],
        "label": contrast["label"],
        "n_subjects": len(subjects),
        "mean_rho_magnitude": per_subject["rho_magnitude"].mean(),
        "sd_rho_magnitude": per_subject["rho_magnitude"].std(ddof=1),
        "mean_noise_ceiling": per_subject["noise_ceiling"].mean(),
        "mean_pct_of_ceiling": per_subject["pct_of_ceiling"].mean(),
        "n_sig_magnitude": int((per_subject["p_magnitude_fdr"] < stats_cfg["alpha"]).sum()),
        "legacy_group_rho": rho_group,
        "legacy_group_p": p_group,
        "marker_group_rho": rho_marker,
        "marker_group_p": p_marker,
        "n_sig_markers": int((per_marker["p_across_subjects_fdr"] < stats_cfg["alpha"]).sum()),
        "mean_sign_concordance": float(per_marker["sign_concordance"].mean()),
        "n_sig_sign_markers": int((per_marker["p_sign_concordance_fdr"] < stats_cfg["alpha"]).sum()),
    }
    return (per_subject, per_marker, per_subject_direction, per_feature,
            per_subject_feature_direction, summary)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Entry point: run every configured contrast and write the result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_feature_consistency.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = REPO_ROOT / config["paths"]["results_root"]
    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config["statistics"]["random_state"])

    subject_frames, marker_frames, direction_frames = [], [], []
    feature_frames, feature_direction_frames, summaries = [], [], []
    for contrast in config["contrasts"]:
        print(f"\n=== {contrast['label']} ===", flush=True)
        (per_subject, per_marker, per_subject_direction, per_feature,
         per_subject_feature_direction, summary) = analyse_contrast(
            contrast, config, results_root, rng
        )
        subject_frames.append(per_subject)
        marker_frames.append(per_marker)
        direction_frames.append(per_subject_direction)
        feature_frames.append(per_feature)
        feature_direction_frames.append(per_subject_feature_direction)
        summaries.append(summary)
        print(
            f"  n={summary['n_subjects']}  "
            f"rho_mag={summary['mean_rho_magnitude']:.3f} "
            f"({summary['n_sig_magnitude']}/{summary['n_subjects']} sig)  "
            f"ceiling={summary['mean_noise_ceiling']:.3f}  "
            f"{summary['mean_pct_of_ceiling']:.0f}% of ceiling  |  "
            f"group rho: feature={summary['legacy_group_rho']:.3f} "
            f"marker={summary['marker_group_rho']:.3f}  |  "
            f"sign concordance={summary['mean_sign_concordance']:.2f} "
            f"({summary['n_sig_sign_markers']}/{len(per_marker)} markers)",
            flush=True,
        )

    per_subject_all = pd.concat(subject_frames, ignore_index=True)
    per_marker_all = pd.concat(marker_frames, ignore_index=True)
    per_direction_all = pd.concat(direction_frames, ignore_index=True)
    per_feature_all = pd.concat(feature_frames, ignore_index=True)
    per_feature_direction_all = pd.concat(feature_direction_frames, ignore_index=True)
    summary_all = pd.DataFrame(summaries)

    per_subject_all.to_csv(output_dir / "per_subject_consistency.csv", index=False)
    per_marker_all.to_csv(output_dir / "marker_level_consistency.csv", index=False)
    per_direction_all.to_csv(output_dir / "marker_direction_per_subject.csv", index=False)
    per_feature_all.to_csv(output_dir / "feature_level_direction.csv", index=False)
    per_feature_direction_all.to_csv(
        output_dir / "feature_direction_per_subject.csv", index=False)
    summary_all.to_csv(output_dir / "group_summary.csv", index=False)
    summary_all[[
        "contrast", "label", "n_subjects", "mean_rho_magnitude",
        "mean_noise_ceiling", "mean_pct_of_ceiling", "legacy_group_rho",
    ]].to_csv(output_dir / "noise_ceiling_summary.csv", index=False)
    shutil.copy(args.config, output_dir / "used_config.yaml")

    print(f"\nWrote 7 tables + used_config.yaml to {output_dir}")


if __name__ == "__main__":
    main()
