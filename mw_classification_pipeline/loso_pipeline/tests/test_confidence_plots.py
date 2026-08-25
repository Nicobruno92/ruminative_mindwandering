"""
Tests for confidence column propagation through the prediction pipeline
and for the confidence-specific probability-vs-raw plots.

Covers:
- _save_probabilities writes 'confidence' to sample_predictions.csv
- _consolidate_sample_predictions produces 'confidence_first' column
- plot_probability_vs_raw generates both onoff and confidence plot files
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.analysis_utils import _save_probabilities, _consolidate_sample_predictions
from utils.plotting_utils import plot_probability_vs_raw


# =============================================================================
# Helpers
# =============================================================================

def _make_run_results_df(df: pd.DataFrame, subject: str = "02") -> pd.DataFrame:
    """Build a minimal run_results DataFrame mimicking pipeline output."""
    n = len(df)
    fold_details = [{
        "fold_idx": 0,
        "subject": subject,
        "test_indices": list(range(n)),
        "y_true": [1] * n,
        "y_pred": [1] * n,
        "y_proba": [0.7] * n,
        "label_percentages": {},
    }]
    return pd.DataFrame([{"fold_details": fold_details}])


def _make_df_with_confidence(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject": ["02"] * n,
        "task": ["Sart1"] * n,
        "probe_number": list(range(n)),
        "onoff": rng.integers(0, 101, n).tolist(),
        "confidence": rng.integers(0, 101, n).tolist(),
    })


# =============================================================================
# _save_probabilities
# =============================================================================

class TestSaveProbabilitiesConfidence:

    def test_confidence_column_written(self, tmp_path):
        df = _make_df_with_confidence()
        run_results = _make_run_results_df(df)
        _save_probabilities(run_results, df, str(tmp_path), "test_run", run_idx=0)

        csv = tmp_path / "test_run_sample_predictions.csv"
        assert csv.exists(), "sample_predictions.csv not written"
        saved = pd.read_csv(csv)
        assert "confidence" in saved.columns, "confidence column missing from sample_predictions.csv"

    def test_confidence_values_match_source(self, tmp_path):
        df = _make_df_with_confidence()
        run_results = _make_run_results_df(df)
        _save_probabilities(run_results, df, str(tmp_path), "test_run", run_idx=0)

        saved = pd.read_csv(tmp_path / "test_run_sample_predictions.csv")
        for _, row in saved.iterrows():
            idx = int(row["sample_idx"])
            assert row["confidence"] == df.iloc[idx]["confidence"]

    def test_no_confidence_column_still_works(self, tmp_path):
        """Pipeline must not crash when df lacks confidence."""
        df = _make_df_with_confidence().drop(columns=["confidence"])
        run_results = _make_run_results_df(df)
        _save_probabilities(run_results, df, str(tmp_path), "test_run", run_idx=0)

        saved = pd.read_csv(tmp_path / "test_run_sample_predictions.csv")
        # confidence absent from source → stored as empty string, column still present
        assert "confidence" in saved.columns


# =============================================================================
# _consolidate_sample_predictions
# =============================================================================

class TestConsolidatePredictionsConfidence:

    def _write_run(self, run_dir: Path, run_idx: int, n: int, seed: int):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "run_idx": run_idx,
            "fold_idx": 0,
            "subject": "02",
            "sample_idx": list(range(n)),
            "task": "Sart1",
            "probe_number": list(range(n)),
            "onoff": rng.integers(0, 101, n).tolist(),
            "confidence": rng.integers(0, 101, n).tolist(),
            "y_true": [1] * n,
            "y_pred": [1] * n,
            "y_proba": rng.uniform(0.4, 0.9, n).tolist(),
        })
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "test_base_sample_predictions.csv", index=False)

    def test_confidence_first_column_present(self, tmp_path):
        true_runs = tmp_path / "true_runs"
        for i in range(3):
            self._write_run(true_runs / f"run_{i}", run_idx=i, n=20, seed=i)

        _consolidate_sample_predictions(str(true_runs), "test_base", output_dir=str(tmp_path))

        consolidated = pd.read_csv(tmp_path / "test_base_consolidated_sample_predictions.csv")
        assert "confidence_first" in consolidated.columns, (
            "confidence_first column missing from consolidated_sample_predictions.csv"
        )

    def test_confidence_first_values_are_numeric(self, tmp_path):
        true_runs = tmp_path / "true_runs"
        for i in range(2):
            self._write_run(true_runs / f"run_{i}", run_idx=i, n=15, seed=i + 10)

        _consolidate_sample_predictions(str(true_runs), "test_base", output_dir=str(tmp_path))

        consolidated = pd.read_csv(tmp_path / "test_base_consolidated_sample_predictions.csv")
        assert consolidated["confidence_first"].notna().all()
        assert pd.to_numeric(consolidated["confidence_first"], errors="coerce").notna().all()

    def test_without_confidence_still_consolidates(self, tmp_path):
        """Runs without a confidence column should consolidate without errors."""
        true_runs = tmp_path / "true_runs"
        for i in range(2):
            rng = np.random.default_rng(i)
            n = 15
            df = pd.DataFrame({
                "run_idx": i, "fold_idx": 0, "subject": "02",
                "sample_idx": list(range(n)), "task": "Sart1",
                "probe_number": list(range(n)),
                "onoff": rng.integers(0, 101, n).tolist(),
                "y_true": [1] * n, "y_pred": [1] * n,
                "y_proba": rng.uniform(0.4, 0.9, n).tolist(),
            })
            run_dir = true_runs / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(run_dir / "test_base_sample_predictions.csv", index=False)

        _consolidate_sample_predictions(str(true_runs), "test_base", output_dir=str(tmp_path))

        consolidated = pd.read_csv(tmp_path / "test_base_consolidated_sample_predictions.csv")
        assert "onoff_first" in consolidated.columns
        assert "confidence_first" not in consolidated.columns


# =============================================================================
# plot_probability_vs_raw
# =============================================================================

class TestPlotProbabilityVsRaw:

    def _make_consolidated_df(self, n: int = 40, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "subject": [f"{i % 5 + 2:02d}" for i in range(n)],
            "task": "Sart1",
            "probe_number": list(range(n)),
            "y_true_first": rng.integers(0, 2, n).tolist(),
            "onoff_first": rng.integers(0, 101, n).tolist(),
            "confidence_first": rng.integers(0, 101, n).tolist(),
            "proba_mean": rng.uniform(0.3, 0.8, n).tolist(),
            "proba_std": rng.uniform(0.01, 0.1, n).tolist(),
            "pred_proportion": rng.uniform(0.3, 0.7, n).tolist(),
            "n_runs": 10,
            "y_pred_avg": rng.integers(0, 2, n).tolist(),
        })

    def test_generates_onoff_plots(self, tmp_path):
        df = self._make_consolidated_df()
        plot_probability_vs_raw(df, str(tmp_path), "rf_loso_10runs")

        plots_dir = tmp_path / "plots"
        assert (plots_dir / "rf_loso_10runs_prob_vs_onoff_general.png").exists()
        assert (plots_dir / "rf_loso_10runs_prob_vs_onoff_faceted.png").exists()

    def test_generates_confidence_plots(self, tmp_path):
        df = self._make_consolidated_df()
        plot_probability_vs_raw(df, str(tmp_path), "rf_loso_10runs")

        plots_dir = tmp_path / "plots"
        assert (plots_dir / "rf_loso_10runs_prob_vs_confidence_general.png").exists()
        assert (plots_dir / "rf_loso_10runs_prob_vs_confidence_faceted.png").exists()

    def test_no_confidence_col_still_generates_onoff(self, tmp_path):
        df = self._make_consolidated_df().drop(columns=["confidence_first"])
        plot_probability_vs_raw(df, str(tmp_path), "rf_loso_10runs")

        plots_dir = tmp_path / "plots"
        assert (plots_dir / "rf_loso_10runs_prob_vs_onoff_general.png").exists()
        assert not (plots_dir / "rf_loso_10runs_prob_vs_confidence_general.png").exists()

    def test_no_raw_cols_prints_warning(self, tmp_path, capsys):
        df = pd.DataFrame({
            "subject": ["02"] * 5,
            "y_true_first": [1] * 5,
            "proba_mean": [0.6] * 5,
        })
        plot_probability_vs_raw(df, str(tmp_path), "rf_loso_10runs")
        captured = capsys.readouterr()
        assert "No raw dimension column" in captured.out
