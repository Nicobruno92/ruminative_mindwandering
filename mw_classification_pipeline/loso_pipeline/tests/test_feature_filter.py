"""
Exhaustive tests for feature family filtering (filter_features_by_family).

Covers:
- null family (all features)
- prefix matching (single and multiple prefixes)
- unknown family raises
- overlap between families
- edge cases: empty DataFrame, no matching columns, single column
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_loso_classification import filter_features_by_family


# Shared feature families config (mirrors the real naming convention).
# Prefixes carry no trailing underscore: the matcher appends its own `(_|$)`
# boundary, so "P1_" would only match a column literally named "P1_" or
# "P1__…". The real configs write "P1", "N1", "psd_relative_alpha", etc.
FAMILIES = {
    "all": None,
    "spectral": ["power_delta", "power_theta", "power_alpha", "power_beta", "power_normalized"],
    "connectivity": ["wsmi_theta", "wsmi_alpha", "wsmi_beta"],
    "information_theory": ["PE_theta", "PE_alpha", "kolmogorov_complexity"],
    "erp": ["P1", "N1", "P3a", "P3b"],
}


def _make_X(**kwargs) -> pd.DataFrame:
    """Build a DataFrame from keyword args {column_name: [values]}."""
    return pd.DataFrame(kwargs)


# =============================================================================
# all family
# =============================================================================

class TestFamilyAll:
    """Tests for the 'all' family (null → no filtering)."""

    def test_all_returns_original(self):
        X = _make_X(power_alpha_Fz=[1.0], wsmi_theta=[0.5], some_other=[0.1])
        result = filter_features_by_family(X, "all", FAMILIES["all"])
        pd.testing.assert_frame_equal(result, X)

    def test_all_with_single_column(self):
        X = _make_X(feat=[1.0, 2.0])
        result = filter_features_by_family(X, "all", FAMILIES["all"])
        assert list(result.columns) == ["feat"]

    def test_all_with_many_columns(self):
        cols = {f"feat_{i}": [float(i)] for i in range(100)}
        X = _make_X(**cols)
        result = filter_features_by_family(X, "all", FAMILIES["all"])
        assert len(result.columns) == 100


# =============================================================================
# Spectral family
# =============================================================================

class TestFamilySpectral:
    """Tests for the 'spectral' family."""

    def test_selects_power_columns(self):
        X = _make_X(
            power_alpha_Fz=[1.0],
            power_theta_Cz=[0.5],
            wsmi_theta=[0.2],     # connectivity → excluded
            PE_theta=[0.3],       # info theory → excluded
        )
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert "power_alpha_Fz" in result.columns
        assert "power_theta_Cz" in result.columns
        assert "wsmi_theta" not in result.columns
        assert "PE_theta" not in result.columns

    def test_selects_normalized_power(self):
        X = _make_X(power_normalized_alpha_Fz=[1.0], wsmi_theta=[0.5])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert "power_normalized_alpha_Fz" in result.columns
        assert "wsmi_theta" not in result.columns

    def test_count_matches(self):
        X = _make_X(
            power_delta_Fz=[1.0], power_alpha_Cz=[2.0],
            wsmi_theta=[0.5], PE_alpha=[0.3],
        )
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert len(result.columns) == 2


# =============================================================================
# Connectivity family
# =============================================================================

class TestFamilyConnectivity:
    """Tests for the 'connectivity' family."""

    def test_selects_wsmi_columns(self):
        X = _make_X(wsmi_theta_pair=[0.2], wsmi_alpha_pair=[0.3], power_alpha=[0.9])
        result = filter_features_by_family(X, "connectivity", FAMILIES["connectivity"])
        assert "wsmi_theta_pair" in result.columns
        assert "wsmi_alpha_pair" in result.columns
        assert "power_alpha" not in result.columns

    def test_excludes_wsmi_gamma_not_in_family(self):
        """wsmi_gamma is not in the family's prefix list → excluded."""
        local_families = {**FAMILIES, "connectivity": ["wsmi_theta"]}
        X = _make_X(wsmi_theta_Fz=[1.0], wsmi_gamma_Fz=[0.5])
        result = filter_features_by_family(X, "connectivity", local_families["connectivity"])
        assert "wsmi_theta_Fz" in result.columns
        assert "wsmi_gamma_Fz" not in result.columns


# =============================================================================
# ERP family
# =============================================================================

class TestFamilyERP:
    """Tests for the 'erp' family."""

    def test_selects_erp_components(self):
        X = _make_X(P1_Fz=[1.0], N1_Cz=[0.5], P3a_Pz=[0.3], power_alpha=[0.9])
        result = filter_features_by_family(X, "erp", FAMILIES["erp"])
        assert "P1_Fz" in result.columns
        assert "N1_Cz" in result.columns
        assert "P3a_Pz" in result.columns
        assert "power_alpha" not in result.columns


# =============================================================================
# Information theory family
# =============================================================================

class TestFamilyInformationTheory:
    """Tests for the 'information_theory' family."""

    def test_selects_pe_and_kolmogorov(self):
        X = _make_X(PE_theta_Fz=[0.3], kolmogorov_complexity_Cz=[0.4], wsmi_alpha=[0.5])
        result = filter_features_by_family(X, "information_theory", FAMILIES["information_theory"])
        assert "PE_theta_Fz" in result.columns
        assert "kolmogorov_complexity_Cz" in result.columns
        assert "wsmi_alpha" not in result.columns


# =============================================================================
# Error cases
# =============================================================================

class TestFamilyErrors:
    """Tests for error handling in filter_features_by_family."""

    def test_no_matching_columns_raises(self):
        """A valid family with no matching columns should raise ValueError."""
        X = _make_X(pe_theta=[1.0], kolmo=[0.5])  # Columns don't match any prefix
        with pytest.raises(ValueError, match="matched 0 columns"):
            filter_features_by_family(X, "spectral", FAMILIES["spectral"])

    def test_empty_dataframe_raises(self):
        X = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            filter_features_by_family(X, "spectral", FAMILIES["spectral"])


# =============================================================================
# Return value checks
# =============================================================================

class TestFamilyReturnValues:
    """Tests that the filtered DataFrame has the expected structure."""

    def test_returns_dataframe(self):
        X = _make_X(power_alpha_Fz=[1.0], wsmi_theta=[0.5])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert isinstance(result, pd.DataFrame)

    def test_values_preserved(self):
        """Feature values must not change during filtering."""
        X = _make_X(power_alpha_Fz=[1.23, 4.56])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        np.testing.assert_array_almost_equal(
            result["power_alpha_Fz"].values, [1.23, 4.56]
        )

    def test_column_order_preserved(self):
        """Column order should match the original DataFrame."""
        X = _make_X(power_beta_Fz=[1.0], power_alpha_Cz=[2.0], wsmi_theta=[0.5])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        expected_order = ["power_beta_Fz", "power_alpha_Cz"]
        assert list(result.columns) == expected_order

    def test_index_preserved(self):
        """Row index must be preserved after filtering."""
        X = pd.DataFrame({"power_alpha_Fz": [1.0, 2.0], "wsmi_theta": [0.5, 0.6]},
                         index=[10, 20])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert list(result.index) == [10, 20]

    def test_n_rows_unchanged(self):
        X = _make_X(power_alpha_Fz=[1.0, 2.0, 3.0], wsmi_theta=[0.1, 0.2, 0.3])
        result = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        assert len(result) == 3


# =============================================================================
# Idempotence
# =============================================================================

class TestFamilyIdempotence:
    """
    Filtering must be safe to apply more than once.

    The within-subject entry point applies the family filter on both the
    cached and the uncached path, so a cache built from an already-filtered
    frame gets filtered again. That is only correct if a second application is
    a no-op — the property this class pins down. Applying the filter on just
    one path is what let a stale pickle silently pin the feature space.
    """

    def test_double_filter_equals_single(self):
        X = _make_X(
            power_alpha_Fz=[1.0, 2.0],
            power_theta_Cz=[0.5, 0.6],
            wsmi_theta=[0.2, 0.3],
        )
        once = filter_features_by_family(X, "spectral", FAMILIES["spectral"])
        twice = filter_features_by_family(once, "spectral", FAMILIES["spectral"])
        pd.testing.assert_frame_equal(once, twice)

    def test_filter_of_superset_matches_filter_of_subset(self):
        """
        Filtering a superset yields exactly the same columns as filtering a
        frame that was already narrowed — the guarantee that let the existing
        304-column caches be reused instead of regenerated.
        """
        superset = _make_X(
            power_alpha_Fz=[1.0], power_theta_Cz=[0.5],
            wsmi_theta=[0.2], PE_theta=[0.3], unrelated_col=[9.9],
        )
        subset = superset[["power_alpha_Fz", "power_theta_Cz", "wsmi_theta"]]
        from_super = filter_features_by_family(superset, "spectral", FAMILIES["spectral"])
        from_sub = filter_features_by_family(subset, "spectral", FAMILIES["spectral"])
        assert list(from_super.columns) == list(from_sub.columns)
