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
