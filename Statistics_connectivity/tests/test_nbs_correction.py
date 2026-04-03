"""
Tests for the NBS (Network-Based Statistic) correction module.

Covers:
- _extract_unique_nodes: correct node extraction from connection IDs
- _find_supra_threshold_components: positive/negative separation,
  threshold filtering, correct edge lists
- apply_nbs_correction: Type I error control (pure null) and
  statistical power (known connected component)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from lmm_connectivity import (
    _extract_unique_nodes,
    _find_supra_threshold_components,
    apply_nbs_correction,
    run_lmm_per_connection,
    PAIR_SEPARATOR,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_conn_id(a: str, b: str) -> str:
    return f"{a}{PAIR_SEPARATOR}{b}"


@pytest.fixture
def four_node_t_values():
    """
    A simple four-node graph (A, B, C, D) with known t-statistics.

    Positive supra-threshold (|t|>=3): A--B (t=4), B--C (t=3.5)  → one component
    Negative supra-threshold (|t|>=3): C--D (t=-4)                → isolated edge
    Sub-threshold: A--D (t=1.5)                                   → not included
    """
    return {
        _make_conn_id("A", "B"): 4.0,
        _make_conn_id("B", "C"): 3.5,
        _make_conn_id("C", "D"): -4.0,
        _make_conn_id("A", "D"): 1.5,
    }


@pytest.fixture
def four_nodes():
    return {"A", "B", "C", "D"}


# =============================================================================
# _extract_unique_nodes
# =============================================================================

class TestExtractUniqueNodes:

    def test_basic_extraction(self):
        conn_ids = [
            _make_conn_id("Fp1", "F3"),
            _make_conn_id("F3", "C3"),
            _make_conn_id("Fp1", "Fp2"),
        ]
        nodes = _extract_unique_nodes(conn_ids)
        assert nodes == {"Fp1", "F3", "C3", "Fp2"}

    def test_no_duplicates(self):
        conn_ids = [
            _make_conn_id("A", "B"),
            _make_conn_id("A", "C"),
            _make_conn_id("B", "C"),
        ]
        nodes = _extract_unique_nodes(conn_ids)
        assert len(nodes) == 3

    def test_invalid_format_skipped(self):
        """Connections without exactly 2 parts are silently skipped."""
        conn_ids = [
            _make_conn_id("A", "B"),
            "no_separator_here",
        ]
        nodes = _extract_unique_nodes(conn_ids)
        assert nodes == {"A", "B"}

    def test_empty_input(self):
        assert _extract_unique_nodes([]) == set()


# =============================================================================
# _find_supra_threshold_components
# =============================================================================

class TestFindSupraThresholdComponents:

    def test_returns_two_lists(self, four_node_t_values, four_nodes):
        result = _find_supra_threshold_components(
            four_node_t_values, four_nodes, threshold=3.0
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        pos_comps, neg_comps = result
        assert isinstance(pos_comps, list)
        assert isinstance(neg_comps, list)

    def test_positive_component_found(self, four_node_t_values, four_nodes):
        """A--B and B--C (t=4, 3.5) should form one positive component."""
        pos_comps, neg_comps = _find_supra_threshold_components(
            four_node_t_values, four_nodes, threshold=3.0
        )
        assert len(pos_comps) == 1
        pos_edges = set(pos_comps[0])
        assert _make_conn_id("A", "B") in pos_edges
        assert _make_conn_id("B", "C") in pos_edges

    def test_negative_component_found(self, four_node_t_values, four_nodes):
        """C--D (t=-4) should appear as a negative component (single edge)."""
        pos_comps, neg_comps = _find_supra_threshold_components(
            four_node_t_values, four_nodes, threshold=3.0
        )
        assert len(neg_comps) == 1
        neg_edges = set(neg_comps[0])
        assert _make_conn_id("C", "D") in neg_edges

    def test_subthreshold_edge_excluded(self, four_node_t_values, four_nodes):
        """A--D (t=1.5) is below threshold and must not appear in any component."""
        pos_comps, neg_comps = _find_supra_threshold_components(
            four_node_t_values, four_nodes, threshold=3.0
        )
        all_edges = [e for comp in pos_comps + neg_comps for e in comp]
        assert _make_conn_id("A", "D") not in all_edges

    def test_positive_and_negative_never_merged(self, four_nodes):
        """Positive and negative edges sharing a node must not form one component."""
        # A--B positive, A--B negative (two separate t-values)
        # Use different connections sharing node A
        t_values = {
            _make_conn_id("A", "B"): 4.0,   # positive
            _make_conn_id("A", "C"): -4.0,  # negative, shares A with above
        }
        pos_comps, neg_comps = _find_supra_threshold_components(
            t_values, four_nodes, threshold=3.0
        )
        # No component should contain edges from both directions
        for comp in pos_comps:
            for edge in comp:
                assert t_values[edge] > 0, "Positive component contains negative edge"
        for comp in neg_comps:
            for edge in comp:
                assert t_values[edge] < 0, "Negative component contains positive edge"

    def test_no_supra_threshold_returns_empty_lists(self, four_nodes):
        t_values = {
            _make_conn_id("A", "B"): 1.0,
            _make_conn_id("B", "C"): -1.5,
        }
        pos_comps, neg_comps = _find_supra_threshold_components(
            t_values, four_nodes, threshold=3.0
        )
        assert pos_comps == []
        assert neg_comps == []

    def test_nan_t_values_excluded(self, four_nodes):
        t_values = {
            _make_conn_id("A", "B"): np.nan,
            _make_conn_id("B", "C"): 4.0,
        }
        pos_comps, neg_comps = _find_supra_threshold_components(
            t_values, four_nodes, threshold=3.0
        )
        all_edges = [e for comp in pos_comps + neg_comps for e in comp]
        assert _make_conn_id("A", "B") not in all_edges

    def test_high_threshold_finds_nothing(self, four_node_t_values, four_nodes):
        pos_comps, neg_comps = _find_supra_threshold_components(
            four_node_t_values, four_nodes, threshold=100.0
        )
        assert pos_comps == []
        assert neg_comps == []


# =============================================================================
# apply_nbs_correction — Type I error control and statistical power
# =============================================================================

def _generate_connected_component_data(
    n_subjects: int = 20,
    n_probes_per_sub: int = 20,
    effect_size: float = 0.0,
    n_nodes: int = 6,
    signal_connections: list = None,
    random_state: int = 42,
) -> tuple:
    """
    Generate synthetic connectivity data with an optional planted connected component.

    The "channel" network has `n_nodes` nodes forming a ring topology
    (every adjacent pair is a connection).  If `effect_size > 0`, a subset
    of connections listed in `signal_connections` carries a true positive
    effect; all others are pure noise.

    Returns
    -------
    (df_wide, connection_ids, results_df)
        - df_wide: wide-format DataFrame ready for LMM fitting
        - connection_ids: list of all connection column names
        - results_df: pre-fitted LMM results (run_lmm_per_connection)
    """
    rng = np.random.RandomState(random_state)
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]

    # Build all pairwise connections
    all_conns = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            all_conns.append(_make_conn_id(nodes[i], nodes[j]))

    if signal_connections is None:
        signal_connections = []

    data = []
    for sub_idx in range(1, n_subjects + 1):
        subj_intercept = rng.normal(0, 1.0)
        for p in range(n_probes_per_sub):
            onoff = rng.uniform(0, 100)
            row = {
                "subject": f"{sub_idx:02d}",
                "task": "Sart1",
                "probe_number": p + 1,
                "onoff": onoff,
            }
            for conn in all_conns:
                signal = effect_size * (onoff / 100.0) if conn in signal_connections else 0.0
                row[conn] = 0.5 + subj_intercept + signal + rng.normal(0, 0.5)
            data.append(row)

    df_wide = pd.DataFrame(data)
    connection_ids = all_conns

    results_df = run_lmm_per_connection(
        df_wide=df_wide,
        connection_ids=connection_ids,
        formula="power ~ onoff + (1|subject)",
        predictor_of_interest="onoff",
        random_state=random_state,
    )

    return df_wide, connection_ids, results_df


class TestNBSCorrectionTypeI:
    """
    NBS Type I error control: under pure null (no effect), the proportion
    of components declared significant must not systematically exceed alpha.

    We use a single-run assertion rather than a Monte Carlo family of
    experiments, because a MC study would be prohibitively slow.  Instead
    we verify that the p-values for any spurious components are not
    unreasonably small (> alpha margin).  This is a sanity check, not a
    calibration estimate.
    """

    def test_pure_null_no_spurious_significance(self):
        """
        Under the null (effect_size=0), NBS should not find any significant
        component at alpha=0.05 with reasonable permutations.
        """
        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=20,
            n_probes_per_sub=20,
            effect_size=0.0,
            random_state=99,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=500,   # Fast for CI; increase to 2000+ for final runs
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        n_significant = int(corrected["significant_nbs"].sum())
        # Under the null with a reasonable threshold (|t|>=3 ≈ p<0.003 uncorrected)
        # the chance of even forming a supra-threshold component is very low.
        # We allow at most 1 spurious significant edge as a safety margin.
        assert n_significant <= 1, (
            f"NBS declared {n_significant} significant edge(s) under pure null "
            f"(possible Type I inflation)"
        )


class TestNBSCorrectionPower:
    """
    NBS power: a planted connected component with a strong signal must be
    detected at alpha=0.05.
    """

    def test_detects_planted_connected_component(self):
        """
        A 4-edge connected component (A--B, B--C, C--D, D--E) with strong
        effect should be declared significant.
        """
        signal_connections = [
            _make_conn_id("A", "B"),
            _make_conn_id("B", "C"),
            _make_conn_id("C", "D"),
            _make_conn_id("D", "E"),
        ]

        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=25,
            n_probes_per_sub=20,
            effect_size=3.5,   # Strong effect to ensure |t|>=3 threshold is exceeded
            signal_connections=signal_connections,
            random_state=7,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=500,
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        # All planted signal edges should be in a significant NBS component
        sig_edges = set(
            corrected.loc[corrected["significant_nbs"], "connection_id"]
        )
        for conn in signal_connections:
            assert conn in sig_edges, (
                f"NBS failed to detect planted signal edge {conn} "
                f"(possible Type II error / insufficient power)"
            )

    def test_nbs_component_ids_assigned(self):
        """Significant edges must have a valid (>= 0) nbs_component_id."""
        signal_connections = [
            _make_conn_id("A", "B"),
            _make_conn_id("B", "C"),
            _make_conn_id("C", "D"),
            _make_conn_id("D", "E"),
        ]

        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=25,
            n_probes_per_sub=20,
            effect_size=3.5,
            signal_connections=signal_connections,
            random_state=7,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=500,
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        sig_rows = corrected[corrected["significant_nbs"]]
        if len(sig_rows) > 0:
            assert (sig_rows["nbs_component_id"] >= 0).all(), (
                "Significant edges have invalid component_id=-1"
            )

    def test_direction_separation_positive_effect(self):
        """
        A planted positive-effect component should yield significant_nbs=True
        and the component_id should not mix with negative-effect connections.
        """
        signal_connections = [
            _make_conn_id("A", "B"),
            _make_conn_id("B", "C"),
            _make_conn_id("A", "C"),
        ]

        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=25,
            n_probes_per_sub=20,
            effect_size=3.5,
            signal_connections=signal_connections,
            random_state=13,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=500,
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        # Signal connections should be positive (positive effect = positive t-stat)
        for conn in signal_connections:
            row = corrected[corrected["connection_id"] == conn].iloc[0]
            assert row["t_statistic"] > 0, (
                f"Planted positive-effect edge {conn} has negative t-stat"
            )
            assert row["significant_nbs"], (
                f"Planted positive-effect edge {conn} not detected by NBS"
            )


class TestNBSPValues:
    """Sanity checks on NBS p-value structure."""

    def test_pvalues_in_valid_range(self):
        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=20,
            n_probes_per_sub=15,
            effect_size=0.0,
            random_state=55,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=200,
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        valid_pvals = corrected["p_value_nbs"].dropna()
        if len(valid_pvals) > 0:
            # Phipson & Smyth correction guarantees p >= 1/(n_perm+1)
            min_possible = 1 / (200 + 1)
            assert (valid_pvals >= min_possible - 1e-10).all(), (
                "NBS p-values below Phipson & Smyth minimum"
            )
            assert (valid_pvals <= 1.0 + 1e-10).all(), (
                "NBS p-values exceed 1.0"
            )

    def test_same_component_same_pvalue(self):
        """All edges in the same NBS component must share the same p-value."""
        signal_connections = [
            _make_conn_id("A", "B"),
            _make_conn_id("B", "C"),
            _make_conn_id("C", "D"),
            _make_conn_id("D", "E"),
        ]

        df_wide, connection_ids, results_df = _generate_connected_component_data(
            n_subjects=25,
            n_probes_per_sub=20,
            effect_size=3.5,
            signal_connections=signal_connections,
            random_state=7,
        )

        corrected = apply_nbs_correction(
            results_df=results_df,
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            primary_threshold=3.0,
            n_permutations=300,
            alpha=0.05,
            random_state=42,
            n_jobs=1,
        )

        for comp_id in corrected["nbs_component_id"].unique():
            if comp_id == -1:
                continue
            comp_rows = corrected[corrected["nbs_component_id"] == comp_id]
            pvals = comp_rows["p_value_nbs"].dropna().unique()
            assert len(pvals) == 1, (
                f"Component {comp_id} has {len(pvals)} different p-values; "
                f"all edges in one component must share the same p-value"
            )
