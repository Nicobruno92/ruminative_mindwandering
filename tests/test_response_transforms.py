"""
Tests for the per-marker response transform used by the Andrillon pipeline.

Two properties matter scientifically and are asserted here:

1. The transform must actually fix residual shape, and must be applied *before*
   the per-subject z-score — applying it after would be a no-op, because the
   z-score is linear and cannot change skewness.
2. Which marker gets which transform must come from configuration only, so the
   choice is fixed before any cluster p-value is seen. A marker absent from the
   config map must fall through to the declared default, never to a guess.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

_STATS_DIR = Path(__file__).resolve().parents[1] / "Statistics"
sys.path.insert(0, str(_STATS_DIR))

from helpers import (  # noqa: E402
    SUPPORTED_RESPONSE_TRANSFORMS,
    apply_response_transform,
    normalize_by_subject,
    resolve_response_transform,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_SUBJECTS = 8
N_PROBES_PER_SUBJECT = 25
N_CHANNELS = 4
SEED = 42

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def behavioral() -> pd.DataFrame:
    """Balanced subject × probe frame matching the synthetic power matrices."""
    subjects = np.repeat(
        [f"{i:02d}" for i in range(2, 2 + N_SUBJECTS)], N_PROBES_PER_SUBJECT
    )
    return pd.DataFrame({"subject": subjects})


@pytest.fixture
def skewed_power() -> np.ndarray:
    """Strictly positive, strongly right-skewed marker values."""
    rng = np.random.default_rng(SEED)
    return rng.lognormal(
        mean=0.0, sigma=1.2,
        size=(N_SUBJECTS * N_PROBES_PER_SUBJECT, N_CHANNELS),
    )


# =============================================================================
# TESTS — transform behaviour
# =============================================================================


def test_log_removes_skew(skewed_power):
    """A log transform must substantially reduce skewness of a lognormal."""
    before = stats.skew(skewed_power.ravel())
    after = stats.skew(apply_response_transform(skewed_power, "log").ravel())

    assert before > 2.0
    assert abs(after) < 0.2


def test_zscore_alone_cannot_fix_skew(skewed_power, behavioral):
    """The per-subject z-score is a linear map and cannot remove skewness.

    This is the reason the transform is applied before normalization rather than
    after, so it is asserted rather than left as a comment. Within a subject the
    map is exactly linear, so that subject's skewness is preserved to numerical
    precision. Pooled across subjects the value shifts somewhat — mixing subjects
    whose scales differ is not itself a linear map — but stays far from symmetric,
    which is the property that matters.
    """
    normalized = normalize_by_subject(
        power_data=skewed_power,
        df_behavioral=behavioral,
        method="zscore",
        channel_wise=False,
        verbose=False,
    )

    subjects = behavioral["subject"].to_numpy()
    for subject in pd.unique(subjects):
        mask = subjects == subject
        assert stats.skew(normalized[mask, :].ravel()) == pytest.approx(
            stats.skew(skewed_power[mask, :].ravel()), rel=1e-6
        )

    # And pooled: still heavily skewed, i.e. normalization solved nothing.
    assert stats.skew(normalized.ravel()) > 3.0


def test_transform_then_normalize_beats_normalize_only(skewed_power, behavioral):
    """Transform-then-z-score must end up far less skewed than z-score alone."""
    normalize_only = normalize_by_subject(
        power_data=skewed_power, df_behavioral=behavioral,
        method="zscore", channel_wise=False, verbose=False,
    )
    transform_then_normalize = normalize_by_subject(
        power_data=apply_response_transform(skewed_power, "log"),
        df_behavioral=behavioral,
        method="zscore", channel_wise=False, verbose=False,
    )

    assert abs(stats.skew(transform_then_normalize.ravel())) < abs(
        stats.skew(normalize_only.ravel())
    )
    assert abs(stats.skew(transform_then_normalize.ravel())) < 0.2


def test_log1p_accepts_zeros_and_log_rejects_them():
    """Count-like data with exact zeros must route to log1p, and log must refuse."""
    counts = np.array([[0.0, 3.0], [7.0, 0.0], [2.0, 5.0]])

    transformed = apply_response_transform(counts, "log1p")
    assert np.all(np.isfinite(transformed))
    assert transformed[0, 0] == 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        apply_response_transform(counts, "log")


def test_logit_rejects_out_of_range_values():
    """Values outside [0, 1] must raise rather than silently produce NaN."""
    proportions = np.array([[0.2, 0.5], [0.9, 1.4]])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_response_transform(proportions, "logit")


def test_logit_keeps_boundary_values_finite():
    """Exact 0 and 1 must survive as finite values, not drop the channel."""
    proportions = np.array([[0.0, 0.5], [1.0, 0.25]])

    transformed = apply_response_transform(proportions, "logit")

    assert np.all(np.isfinite(transformed))


def test_nans_are_preserved(skewed_power):
    """Missing observations must stay missing, in place."""
    power = skewed_power.copy()
    power[3, 1] = np.nan

    transformed = apply_response_transform(power, "log")

    assert np.isnan(transformed[3, 1])
    assert np.isfinite(transformed[4, 1])


def test_rank_inverse_normal_is_gaussian_within_subject(skewed_power, behavioral):
    """Rank-INT must produce near-Gaussian values regardless of input shape."""
    transformed = apply_response_transform(
        skewed_power, "rank_inverse_normal", df_behavioral=behavioral
    )

    assert abs(stats.skew(transformed.ravel())) < 0.1
    assert abs(stats.kurtosis(transformed.ravel())) < 0.3


def test_rank_inverse_normal_requires_behavioral(skewed_power):
    """Without the subject frame the transform must refuse, not rank globally."""
    with pytest.raises(ValueError, match="df_behavioral"):
        apply_response_transform(skewed_power, "rank_inverse_normal")


def test_none_is_identity(skewed_power):
    """'none' must return the values unchanged."""
    np.testing.assert_array_equal(
        apply_response_transform(skewed_power, "none"), skewed_power
    )


def test_unsupported_transform_raises(skewed_power):
    """An unknown transform name must fail loudly."""
    with pytest.raises(ValueError, match="Unsupported response transform"):
        apply_response_transform(skewed_power, "boxcox")


# =============================================================================
# TESTS — config resolution
# =============================================================================


def test_disabled_section_yields_none():
    """With the section disabled, every marker must resolve to 'none'."""
    config = {
        "preprocessing": {
            "response_transform": {
                "enabled": False,
                "default": "none",
                "by_marker": {"sleep/psd_relative_gamma": "log"},
            }
        }
    }

    assert resolve_response_transform(config, "sleep/psd_relative_gamma") == "none"


def test_missing_section_yields_none():
    """A config without the section at all must not crash and must return 'none'."""
    assert resolve_response_transform({"preprocessing": {}}, "sleep/wsmi_beta") == "none"
    assert resolve_response_transform({}, "sleep/wsmi_beta") == "none"


def test_by_marker_map_is_respected():
    """Listed markers get their transform; unlisted ones fall to the default."""
    config = {
        "preprocessing": {
            "response_transform": {
                "enabled": True,
                "default": "none",
                "by_marker": {
                    "sleep/psd_relative_gamma": "log",
                    "sleep/slowwaves_Density": "log1p",
                },
            }
        }
    }

    assert resolve_response_transform(config, "sleep/psd_relative_gamma") == "log"
    assert resolve_response_transform(config, "sleep/slowwaves_Density") == "log1p"
    assert resolve_response_transform(config, "sleep/wsmi_beta") == "none"
    assert resolve_response_transform(config, "evoked/P3b") == "none"


def test_unsupported_transform_in_config_raises():
    """A typo in the config must abort the run, not fall back to no transform."""
    config = {
        "preprocessing": {
            "response_transform": {
                "enabled": True,
                "default": "none",
                "by_marker": {"sleep/psd_relative_gamma": "lgo"},
            }
        }
    }

    with pytest.raises(ValueError, match="not supported"):
        resolve_response_transform(config, "sleep/psd_relative_gamma")


def test_project_config_transforms_resolve():
    """The committed config must declare only supported transforms."""
    import yaml

    config_path = (
        Path(__file__).resolve().parents[1]
        / "Stats_andrillon" / "config_andrillon.yaml"
    )
    config = yaml.safe_load(open(config_path))
    section = config["preprocessing"]["response_transform"]

    for marker, transform in (section.get("by_marker") or {}).items():
        assert transform in SUPPORTED_RESPONSE_TRANSFORMS, marker
    assert section["default"] in SUPPORTED_RESPONSE_TRANSFORMS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
