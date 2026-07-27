"""
Unit tests for the family-level (omnibus) permutation test.

The critical property is calibration: under the global null the omnibus
p-value must be approximately uniform, so that a nominal 5% test rejects
about 5% of the time. Rank/tie handling is the easiest thing to get subtly
wrong here, and a miscalibrated omnibus would silently manufacture
dimension-level findings.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stats_andrillon"))

import pandas as pd  # noqa: E402

from omnibus_test import (  # noqa: E402
    apply_cross_dimension_correction,
    extract_null_matrices,
    observed_marker_pvalues,
    permutation_marker_pvalues,
    run_omnibus,
)

N_PERM = 500
N_MARKERS = 10


def _make_marker(name, pos, neg, cluster_p=None):
    """Build a minimal marker result dict."""
    result = {
        "marker_name": name,
        "marker_type": "state",
        "null_cluster_stats": {"pos_max": pos, "neg_min": neg},
    }
    if cluster_p is not None:
        result["cluster_p_values"] = np.asarray(cluster_p, dtype=float)
    return result


# =============================================================================
# Rank counting
# =============================================================================

def test_permutation_pvalues_are_in_unit_interval():
    rng = np.random.default_rng(0)
    pos = np.abs(rng.normal(size=(N_MARKERS, N_PERM)))
    neg = -np.abs(rng.normal(size=(N_MARKERS, N_PERM)))
    p = permutation_marker_pvalues(pos, neg)
    assert p.shape == (N_MARKERS, N_PERM)
    assert np.all(p > 0) and np.all(p <= 1)


def test_largest_statistic_gets_smallest_pvalue():
    """The most extreme permutation must rank first."""
    pos = np.array([[1.0, 5.0, 2.0, 3.0]])
    neg = np.zeros((1, 4))
    p = permutation_marker_pvalues(pos, neg)
    # index 1 holds the largest positive statistic
    assert np.argmin(p[0]) == 1
    assert p[0, 1] == pytest.approx(1 / 4)


def test_ties_are_counted_as_equally_extreme():
    """Tied statistics must receive identical p-values, not arbitrary ranks."""
    pos = np.array([[2.0, 2.0, 2.0, 1.0]])
    neg = np.zeros((1, 4))
    p = permutation_marker_pvalues(pos, neg)
    assert p[0, 0] == p[0, 1] == p[0, 2]
    # three of four permutations are >= 2.0
    assert p[0, 0] == pytest.approx(3 / 4)


def test_negative_clusters_are_ranked_by_being_more_negative():
    pos = np.zeros((1, 4))
    neg = np.array([[-1.0, -5.0, -2.0, -3.0]])
    p = permutation_marker_pvalues(pos, neg)
    assert np.argmin(p[0]) == 1


# =============================================================================
# Calibration under the global null
# =============================================================================

def test_omnibus_is_calibrated_under_the_global_null():
    """
    With observed data drawn from the same distribution as the permutations,
    the omnibus p-value must be roughly uniform.

    Implemented by treating one permutation as the observed data and running
    the test on the remainder, repeatedly.
    """
    rng = np.random.default_rng(7)
    alpha = 0.05
    n_trials = 200
    p_counts, p_mins = [], []

    for _ in range(n_trials):
        pos = np.abs(rng.normal(size=(N_MARKERS, N_PERM + 1)))
        neg = -np.abs(rng.normal(size=(N_MARKERS, N_PERM + 1)))

        # last column plays the role of the observed data
        obs_pos, obs_neg = pos[:, -1], neg[:, -1]
        null_pos, null_neg = pos[:, :-1], neg[:, :-1]

        # observed marker p-values against the retained null
        ge = (null_pos >= obs_pos[:, None]).sum(axis=1) + 1
        le = (null_neg <= obs_neg[:, None]).sum(axis=1) + 1
        obs_p = np.minimum(ge, le) / (N_PERM + 1)

        perm_p = permutation_marker_pvalues(null_pos, null_neg)

        t_obs = int(np.sum(obs_p <= alpha))
        t_null = np.sum(perm_p <= alpha, axis=0)
        p_counts.append((np.sum(t_null >= t_obs) + 1) / (N_PERM + 1))

        p_mins.append(
            (np.sum(perm_p.min(axis=0) <= obs_p.min()) + 1) / (N_PERM + 1)
        )

    # A calibrated test rejects at about the nominal rate. The tolerance is
    # wide because n_trials is small; the point is to catch gross
    # miscalibration (e.g. rejecting 30% of the time), not to certify exactness.
    rate_count = np.mean(np.array(p_counts) <= alpha)
    rate_min = np.mean(np.array(p_mins) <= alpha)
    assert rate_count < 0.20, f"count statistic over-rejects: {rate_count:.3f}"
    assert rate_min < 0.20, f"min-p statistic over-rejects: {rate_min:.3f}"


def test_omnibus_detects_a_planted_family_effect():
    """A family where several markers carry real signal must be detected."""
    rng = np.random.default_rng(11)
    pos = np.abs(rng.normal(size=(N_MARKERS, N_PERM)))
    neg = -np.abs(rng.normal(size=(N_MARKERS, N_PERM)))

    results = []
    for i in range(N_MARKERS):
        # five markers get a cluster far beyond anything in their null
        cluster_p = [0.001] if i < 5 else [0.6]
        results.append(_make_marker(f"state/m{i}", pos[i], neg[i], cluster_p))

    stats = run_omnibus(results, alpha=0.05)
    assert stats["count_observed"] == 5
    assert stats["count_p_omnibus"] < 0.05
    assert stats["min_p_fwer"] < 0.05


# =============================================================================
# Cross-dimension correction
# =============================================================================

def test_cross_dimension_bh_is_monotone_and_per_family():
    """BH correction runs within each family and never lowers a p-value.

    Uses the real six-dimension sleep p-values, where two dimensions are far
    from chance and three are marginal (~0.04). BH must keep the strong two
    significant and push the marginal three just past 0.05 — the whole point of
    adding the cross-dimension layer.
    """
    dims = ["onoff", "valence_sq", "valence", "time", "time_sq", "selfother"]
    df = pd.DataFrame({
        "target": dims * 2,
        "family": ["sleep"] * 6 + ["evoked"] * 6,
        "count_p_omnibus": [0.0002, 0.0004, 0.038, 0.041, 0.043, 0.085]
                           + [0.094, 1.0, 1.0, 1.0, 0.033, 0.099],
        "min_p_fwer": [0.0002, 0.0002, 0.162, 0.143, 0.091, 0.159]
                      + [0.025, 0.356, 0.283, 0.612, 0.104, 0.121],
    })
    out = apply_cross_dimension_correction(df, alpha=0.05)

    # corrected never below uncorrected
    assert (out["count_p_bh"] >= out["count_p_omnibus"] - 1e-12).all()
    assert (out["min_p_bh"] >= out["min_p_fwer"] - 1e-12).all()

    sleep = out[out.family == "sleep"].set_index("target")
    # the two strong dimensions survive
    assert sleep.loc["onoff", "count_p_bh"] < 0.05
    assert sleep.loc["valence_sq", "count_p_bh"] < 0.05
    # the three marginal ones are lifted past threshold
    for tgt in ["valence", "time", "time_sq"]:
        assert sleep.loc[tgt, "count_p_bh"] > 0.05

    # correction is per-family: sleep values do not depend on evoked's
    recomputed_sleep = apply_cross_dimension_correction(
        df[df.family == "sleep"].copy(), alpha=0.05
    ).set_index("target")["count_p_bh"]
    for tgt in dims:
        assert sleep.loc[tgt, "count_p_bh"] == pytest.approx(recomputed_sleep.loc[tgt])


def test_cross_dimension_single_dimension_is_noop():
    df = pd.DataFrame({
        "target": ["onoff"],
        "family": ["sleep"],
        "count_p_omnibus": [0.01],
        "min_p_fwer": [0.02],
    })
    out = apply_cross_dimension_correction(df, alpha=0.05)
    assert out["count_p_bh"].iloc[0] == pytest.approx(0.01)
    assert out["min_p_bh"].iloc[0] == pytest.approx(0.02)


# =============================================================================
# Input validation
# =============================================================================

def test_missing_nulls_raise():
    r = {"marker_name": "state/x", "marker_type": "state"}
    with pytest.raises(ValueError, match="null_cluster_stats"):
        extract_null_matrices([r])


def test_mismatched_permutation_counts_raise():
    a = _make_marker("state/a", np.ones(10), -np.ones(10))
    b = _make_marker("state/b", np.ones(9), -np.ones(9))
    with pytest.raises(ValueError, match="disagree on the number of permutations"):
        extract_null_matrices([a, b])


def test_marker_without_clusters_is_represented_by_p_one():
    a = _make_marker("state/a", np.ones(5), -np.ones(5), cluster_p=[])
    b = _make_marker("state/b", np.ones(5), -np.ones(5), cluster_p=[0.02])
    _, _, names = extract_null_matrices([a, b])
    obs = observed_marker_pvalues([a, b], names)
    assert obs[names.index("state/a")] == 1.0
    assert obs[names.index("state/b")] == pytest.approx(0.02)
