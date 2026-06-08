"""
Tests for the cross-decoding generalization matrix (predictions-based).

The matrix is assembled POST-HOC from the per-probe predictions that each
per-dimension classification run already saves (``*_consolidated_sample_predictions.csv``:
columns ``subject, task, probe_number, y_true_first, proba_mean``). Nothing is
retrained and no data is reloaded.

Cell (model M -> dimension D) = AUC of M's out-of-fold scores (``proba_mean``)
against D's binary labels (``y_true_first``), over the probes present in BOTH
runs (intersection of their native trial sets). The diagonal reproduces each
dimension's own AUC.

All tests use synthetic prediction tables — deterministic and fast.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.cross_decoding_utils import (
    cross_decode_from_predictions,
    cross_decode_permutation_test,
    save_cross_decoding_outputs,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _pred_table(pids, labels, probas):
    """Build a minimal consolidated-predictions table."""
    subj = [p.split("_")[0] for p in pids]
    task = [p.split("_")[1] for p in pids]
    probe = [int(p.split("_")[2]) for p in pids]
    return pd.DataFrame({
        "subject": subj, "task": task, "probe_number": probe,
        "y_true_first": labels, "proba_mean": probas,
    })


class TestCrossDecodeFromPredictions:
    """Assemble the AUC matrix from saved per-probe predictions."""

    def _tables(self):
        pids = [f"02_Sart1_{i}" for i in range(20)]
        label_a = [i % 2 for i in range(20)]            # alternating 0/1
        tables = {
            # A: perfect self-prediction
            "A": _pred_table(pids, label_a, [float(v) for v in label_a]),
            # B: identical labels to A, on the same probes
            "B": _pred_table(pids, label_a, np.random.default_rng(0).random(20)),
            # D: inverse labels to A, same probes
            "D": _pred_table(pids, [1 - v for v in label_a], np.random.default_rng(1).random(20)),
            # E: only 5 probes overlap with A -> below min_overlap
            "E": _pred_table([f"02_Sart1_{i}" for i in range(5)],
                             [i % 2 for i in range(5)], np.random.default_rng(2).random(5)),
        }
        return tables

    def test_diagonal_reproduces_self_auc(self):
        result = cross_decode_from_predictions(self._tables(), min_overlap=4)
        assert result["mean"].loc["A", "A"] == pytest.approx(1.0)

    def test_shared_and_inverse_labels(self):
        result = cross_decode_from_predictions(self._tables(), min_overlap=4)
        m = result["mean"]
        # A's perfect score vs B's identical labels -> AUC 1; vs D's inverse -> 0
        assert m.loc["A", "B"] == pytest.approx(1.0)
        assert m.loc["A", "D"] == pytest.approx(0.0)

    def test_small_overlap_is_nan_and_counted(self):
        result = cross_decode_from_predictions(self._tables(), min_overlap=20)
        assert np.isnan(result["mean"].loc["A", "E"])
        assert result["n_overlap"].loc["A", "E"] == 5

    def test_matrix_is_square_over_dimensions(self):
        result = cross_decode_from_predictions(self._tables(), min_overlap=4)
        dims = list(self._tables().keys())
        assert list(result["mean"].index) == dims
        assert list(result["mean"].columns) == dims

    def test_auc_is_averaged_per_subject_not_pooled(self):
        """Each subject ranks its labels perfectly but on a different score scale.

        Pooling all probes would give AUC < 1 (scales overlap badly); averaging
        per subject gives 1.0. The pipeline reports per-subject AUC, so the matrix
        must too — this is what makes the diagonal reproduce the headline AUC.
        """
        pids = ([f"02_Sart1_{i}" for i in range(4)] + [f"03_Sart1_{i}" for i in range(4)])
        labels = [0, 0, 1, 1, 0, 0, 1, 1]
        # subject 02 scores in [0.1,0.4]; subject 03 in [0.5,0.8] — separable
        # within each subject, but pooled the 03 zeros outrank the 02 ones.
        probas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        tables = {"A": _pred_table(pids, labels, probas)}
        result = cross_decode_from_predictions(tables, min_overlap=4)
        assert result["mean"].loc["A", "A"] == pytest.approx(1.0)


class TestCrossDecodePermutationTest:
    """Null that preserves label correlation; isolates signal beyond it."""

    def _tables(self):
        # Two subjects, the 8-probe pattern from the worked example, replicated.
        # m_label (A) and target B correlate; WITHIN each A-class the positive-B
        # probes get the higher scores -> genuine B-specific signal.
        rows_a, rows_b, rows_d2 = [], [], []
        for subj in ("02", "03"):
            a_label = [1, 1, 1, 1, 0, 0, 0, 0]
            b_label = [1, 1, 0, 0, 1, 0, 0, 0]
            score = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
            for i in range(8):
                pid = (subj, "Sart1", i)
                rows_a.append((*pid, a_label[i], score[i]))
                rows_b.append((*pid, b_label[i], 0.5))
                rows_d2.append((*pid, a_label[i], 0.5))  # D2 == A's label (pure correlation)
        cols = ["subject", "task", "probe_number", "y_true_first", "proba_mean"]
        return {
            "A": pd.DataFrame(rows_a, columns=cols),
            "B": pd.DataFrame(rows_b, columns=cols),
            "D2": pd.DataFrame(rows_d2, columns=cols),
        }

    def test_pure_correlation_cell_is_not_significant(self):
        res = cross_decode_permutation_test(self._tables(), n_perm=300, min_overlap=4)
        # A's model perfectly predicts D2 (==A), but ONLY via correlation: within
        # each A-class D2 is constant, so the null reproduces the AUC -> not sig.
        assert res["pvalue"].loc["A", "D2"] > 0.3
        assert res["null_mean"].loc["A", "D2"] == pytest.approx(res["mean"].loc["A", "D2"], abs=0.05)

    def test_genuine_within_class_signal_beats_null(self):
        res = cross_decode_permutation_test(self._tables(), n_perm=300, min_overlap=4)
        # A->B has within-A-class structure -> observed clearly above the null,
        # and more significant than the pure-correlation cell.
        assert res["mean"].loc["A", "B"] - res["null_mean"].loc["A", "B"] > 0.1
        assert res["pvalue"].loc["A", "B"] < res["pvalue"].loc["A", "D2"]


class TestSaveCrossDecodingOutputs:
    """Persisting the matrix as the scientific artifact."""

    def test_mean_csv_roundtrips_and_heatmap_written(self, tmp_path):
        pids = [f"02_Sart1_{i}" for i in range(20)]
        lab = [i % 2 for i in range(20)]
        tables = {
            "A": _pred_table(pids, lab, [float(v) for v in lab]),
            "B": _pred_table(pids, lab, [float(v) for v in lab]),
        }
        result = cross_decode_from_predictions(tables, min_overlap=4)
        save_cross_decoding_outputs(result, str(tmp_path), title="test")
        reloaded = pd.read_csv(tmp_path / "cross_decoding_mean_auc.csv", index_col=0)
        pd.testing.assert_frame_equal(
            reloaded, result["mean"], check_dtype=False, check_names=False
        )
        assert (tmp_path / "cross_decoding_heatmap.png").exists()
