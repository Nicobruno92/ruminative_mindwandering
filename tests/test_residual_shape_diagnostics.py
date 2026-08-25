"""
Tests for the residual-shape diagnostics that replaced Shapiro-Wilk in
``Statistics/lmm_model.run_lmm_per_channel``.

The point of the replacement is that the reported numbers must be *actionable*:
they have to separate a marker whose residuals are benign from one whose
t-statistics could be driven by a handful of extreme probes. These tests build
synthetic data with a known residual shape and assert that the diagnostics
recover it, and that the flag counts follow the documented thresholds.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STATS_DIR = Path(__file__).resolve().parents[1] / "Statistics"
sys.path.insert(0, str(_STATS_DIR))

from lmm_model import (  # noqa: E402
    RESIDUAL_EXKURT_FLAG,
    RESIDUAL_OUTLIER_SD,
    RESIDUAL_SKEW_FLAG,
    run_lmm_per_channel,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_SUBJECTS = 20
N_PROBES_PER_SUBJECT = 30
N_CHANNELS = 3
FORMULA = "power ~ onoff + (1|subject)"
PREDICTOR = "onoff"
SEED = 42

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _make_behavioral(rng: np.random.Generator) -> pd.DataFrame:
    """Build a balanced subject × probe behavioral frame with a numeric predictor."""
    subjects = np.repeat(
        [f"{i:02d}" for i in range(2, 2 + N_SUBJECTS)], N_PROBES_PER_SUBJECT
    )
    return pd.DataFrame({
        "subject": subjects,
        "onoff": rng.uniform(0, 100, size=subjects.size),
    })


def _fit(power_data: np.ndarray, df_behavioral: pd.DataFrame) -> dict:
    """Run the LMM and return only the diagnostics dict."""
    _, _, diagnostics = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=FORMULA,
        predictor_of_interest=PREDICTOR,
        random_state=SEED,
        return_diagnostics=True,
    )
    return diagnostics


# =============================================================================
# TESTS
# =============================================================================


def test_shapiro_keys_are_gone():
    """The removed Shapiro-Wilk keys must not reappear via a partial revert."""
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.normal(size=(len(df), N_CHANNELS))

    diagnostics = _fit(power, df)

    for stale_key in (
        "shapiro_p_values", "shapiro_p_mean",
        "n_normality_violations", "pct_normality_violations",
    ):
        assert stale_key not in diagnostics


def test_gaussian_residuals_are_not_flagged():
    """Clean Gaussian noise must produce near-zero skew and excess kurtosis."""
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.normal(size=(len(df), N_CHANNELS))

    diagnostics = _fit(power, df)

    assert abs(diagnostics["residual_skew_mean"]) < 0.3
    assert abs(diagnostics["residual_exkurt_mean"]) < 0.5
    assert diagnostics["pct_high_skew"] == 0.0
    assert diagnostics["pct_high_kurtosis"] == 0.0
    # A Gaussian puts ~0.006% of mass beyond 4 SD; with n=600 per channel the
    # expected count is well under one observation.
    assert diagnostics["pct_residual_outliers_mean"] < 1.0


def test_summary_is_produced_when_all_fits_carry_warnings():
    """A marker whose channels all converge *with warnings* must still be summarised.

    The summary used to be gated on ``n_converged``, which counts only
    warning-free fits, so exactly those markers silently reported no assumption
    diagnostics at all. Synthetic data with no real between-subject variance
    drives the random-effect variance to its boundary and reproduces that case.
    """
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.normal(size=(len(df), N_CHANNELS))

    diagnostics = _fit(power, df)

    assert diagnostics["n_warnings"] > 0
    assert diagnostics["n_converged"] == 0
    assert "residual_skew_mean" in diagnostics
    assert "pct_residual_outliers_mean" in diagnostics


def test_right_skewed_response_is_detected():
    """A lognormal response must show strong positive skew, above the flag."""
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.lognormal(mean=0.0, sigma=1.0, size=(len(df), N_CHANNELS))

    diagnostics = _fit(power, df)

    assert diagnostics["residual_skew_mean"] > RESIDUAL_SKEW_FLAG
    assert diagnostics["pct_high_skew"] == 100.0


def test_log_transform_removes_the_skew_flag():
    """Log-transforming the same lognormal response must clear the flag.

    This is the property the whole diagnostic exists to support: it has to be
    able to tell a transformed marker from an untransformed one.
    """
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=(len(df), N_CHANNELS))

    diagnostics_raw = _fit(raw, df)
    diagnostics_log = _fit(np.log(raw), df)

    assert diagnostics_raw["pct_high_skew"] == 100.0
    assert diagnostics_log["pct_high_skew"] == 0.0
    assert abs(diagnostics_log["residual_skew_mean"]) < abs(
        diagnostics_raw["residual_skew_mean"]
    )


def test_heavy_tails_are_detected():
    """A t(3) response must show high excess kurtosis and an outlier fraction."""
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.standard_t(df=3, size=(len(df), N_CHANNELS))

    diagnostics = _fit(power, df)

    assert diagnostics["residual_exkurt_mean"] > RESIDUAL_EXKURT_FLAG
    assert diagnostics["pct_high_kurtosis"] > 0.0
    assert diagnostics["pct_residual_outliers_mean"] > 0.0


def test_outlier_percentage_tracks_injected_spikes():
    """Injecting a known number of extreme points must raise the outlier fraction.

    The residuals themselves are not returned, so the assertion is the binding
    weaker property: the fraction becomes positive and cannot exceed the
    proportion of observations that were spiked.
    """
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.normal(size=(len(df), N_CHANNELS))

    n_spikes = 6
    spike_rows = rng.choice(len(df), size=n_spikes, replace=False)
    power[spike_rows, :] += 12.0

    diagnostics = _fit(power, df)

    max_expected_pct = 100.0 * n_spikes / len(df)
    assert 0.0 < diagnostics["pct_residual_outliers_mean"] <= max_expected_pct


def test_per_channel_lists_align_with_fitted_channels():
    """One diagnostic entry per channel that produced a finite t-statistic."""
    rng = np.random.default_rng(SEED)
    df = _make_behavioral(rng)
    power = rng.normal(size=(len(df), N_CHANNELS))

    t_stats, _, diagnostics = run_lmm_per_channel(
        power_data=power,
        df_behavioral=df,
        formula=FORMULA,
        predictor_of_interest=PREDICTOR,
        random_state=SEED,
        return_diagnostics=True,
    )

    n_finite = int(np.sum(np.isfinite(t_stats)))
    assert len(diagnostics["residual_skew"]) == n_finite
    assert len(diagnostics["residual_exkurt"]) == n_finite
    assert len(diagnostics["pct_residual_outliers"]) == n_finite


def test_flag_thresholds_are_exposed_as_constants():
    """Thresholds must be importable constants, not magic numbers in the code."""
    assert RESIDUAL_OUTLIER_SD > 0
    assert RESIDUAL_SKEW_FLAG > 0
    assert RESIDUAL_EXKURT_FLAG > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
