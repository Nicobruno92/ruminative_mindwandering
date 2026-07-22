"""Tests for the GLMM objective-markers backend.

Covers pure response transforms, config loading, the Python-R round trip,
and the count-consistency assumptions the binomial models rely on.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "Behavior" / "Objective_Markers"))

from response_transforms import (  # noqa: E402
    build_binomial_response,
    empirical_logit,
    load_glmm_config,
    log_transform,
)

CONFIG_PATH = REPO_ROOT / "Behavior" / "Objective_Markers" / "glmm_config.yaml"


def test_config_loads_and_has_all_markers():
    cfg = load_glmm_config(CONFIG_PATH)
    families = cfg["markers"]
    assert set(families) == {
        "omission_rate", "commission_rate", "total_errors", "rtcv",
    }
    assert families["omission_rate"]["family"] == "binomial"
    assert families["rtcv"]["family"] == "Gamma"
    assert Path(cfg["r"]["rscript_path"]).exists()


def test_build_binomial_response_adds_success_failure():
    df = pd.DataFrame({"k": [0, 2, 3], "n": [4, 4, 3]})
    out = build_binomial_response(df, "k", "n")
    assert list(out["_succ"]) == [0, 2, 3]
    assert list(out["_fail"]) == [4, 2, 0]


def test_build_binomial_response_drops_zero_denominator():
    df = pd.DataFrame({"k": [0, 1, 0], "n": [0, 1, 2]})
    out = build_binomial_response(df, "k", "n")
    assert len(out) == 2
    assert 0 not in list(out["n"])


def test_build_binomial_response_rejects_numerator_over_denominator():
    df = pd.DataFrame({"k": [5], "n": [4]})
    with pytest.raises(ValueError, match="binomial response is undefined"):
        build_binomial_response(df, "k", "n")


def test_empirical_logit_matches_hand_computation():
    got = empirical_logit(np.array([0, 2]), np.array([4, 4]))
    expected = np.array([
        np.log(0.5 / 4.5),
        np.log(2.5 / 2.5),
    ])
    np.testing.assert_allclose(got, expected)


def test_empirical_logit_finite_at_boundaries():
    got = empirical_logit(np.array([0, 4]), np.array([4, 4]))
    assert np.all(np.isfinite(got))


def test_log_transform_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        log_transform(np.array([1.0, 0.0]))


from glmm_backend import fit_glmm  # noqa: E402

TIDY_COLUMNS = [
    "predictor", "estimate", "std_error", "t_value", "z_value",
    "p_value", "conf_lower", "conf_upper", "p_fdr", "significant_fdr",
]


def _synthetic_binomial(random_state: int = 42) -> pd.DataFrame:
    """Binomial data with a known onoff log-odds effect of 0.8."""
    rng = np.random.default_rng(random_state)
    n_subjects, n_probes, n_trials = 60, 40, 20
    true_beta = 0.8
    rows = []
    for s in range(n_subjects):
        subj_intercept = rng.normal(0.0, 0.5)
        for _ in range(n_probes):
            onoff = rng.normal(0.0, 1.0)
            eta = -1.0 + subj_intercept + true_beta * onoff
            p = 1.0 / (1.0 + np.exp(-eta))
            rows.append({
                "subject": f"S{s:02d}",
                "onoff": onoff,
                "n_events": rng.binomial(n_trials, p),
                "n_total": n_trials,
            })
    return pd.DataFrame(rows)


BINOMIAL_SPEC = {
    "family": "binomial",
    "success_col": "n_events",
    "total_col": "n_total",
}


def test_fit_glmm_recovers_known_effect():
    cfg = load_glmm_config(CONFIG_PATH)
    res = fit_glmm(
        data=_synthetic_binomial(), marker="synthetic", predictors=["onoff"],
        config=cfg, marker_spec=BINOMIAL_SPEC,
    )
    beta = float(res.loc[res["predictor"] == "onoff", "estimate"].iloc[0])
    assert beta == pytest.approx(0.8, abs=0.15), f"recovered {beta}"
    assert bool(res.loc[res["predictor"] == "onoff", "significant_fdr"].iloc[0])


def test_fit_glmm_returns_tidy_schema():
    cfg = load_glmm_config(CONFIG_PATH)
    res = fit_glmm(
        data=_synthetic_binomial(), marker="synthetic", predictors=["onoff"],
        config=cfg, marker_spec=BINOMIAL_SPEC,
    )
    for col in TIDY_COLUMNS:
        assert col in res.columns, f"missing {col}"
    assert bool(res["converged"].iloc[0])
    assert res["dispersion"].iloc[0] > 0
    # t_value is an alias of the Wald z, kept so plotting code needs no change
    np.testing.assert_allclose(res["t_value"].values, res["z_value"].values)
