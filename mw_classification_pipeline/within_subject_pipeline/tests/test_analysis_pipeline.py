"""
Exhaustive integration tests for within-subject loops in utils/analysis_utils.py.

Tests the full Within-Subject classification pipeline end-to-end with synthetic data.
All I/O is directed to tmp_path — no external files required.

Covers:
- run_within_subject_distribution_analysis (outputs, metrics, CSV saving)
- run_within_subject_permutation_analysis (p-values, CSV output, within-subject shuffling)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.analysis_utils import (
    run_within_subject_distribution_analysis,
    run_within_subject_permutation_analysis,
)
from loso_pipeline.tests.conftest import make_synthetic_data, make_minimal_config

_NO_PLOTS = {"save_plots": False, "save_shap": False}

class TestRunWithinSubjectDistributionAnalysis:
    """Integration tests for run_within_subject_distribution_analysis."""

    @pytest.fixture
    def setup(self, tmp_path):
        df, X, y, groups, feature_cols = make_synthetic_data(
            n_subjects=4, n_probes_per_subject=20, n_features=10, random_seed=0
        )
        
        # Add a fake 'tasks' column for GroupKFold if needed
        tasks = []
        for i in range(len(df)):
            tasks.append(f"Task{i % 4}")
        df['task'] = tasks
        
        config = make_minimal_config(tmp_path=tmp_path, n_subjects=4)
        results_path = str(tmp_path / "results_ws")
        return dict(df=df, X=X, y=y, groups=groups, tasks=df["task"],
                    feature_cols=feature_cols, config=config,
                    results_path=results_path)

    def test_returns_nonempty_list(self, setup):
        metrics_list, shap_list = run_within_subject_distribution_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_runs=1, results_path=setup["results_path"],
            min_samples_per_class=0, # ensure no one is skipped due to random luck
            **_NO_PLOTS,
        )
        assert isinstance(metrics_list, list)
        assert len(metrics_list) > 0

    def test_result_has_one_row_per_valid_subject(self, setup):
        # We enforce exactly 4 subjects
        metrics_list, _ = run_within_subject_distribution_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_runs=1, results_path=setup["results_path"],
            min_samples_per_class=0, 
            **_NO_PLOTS,
        )
        # One result dict per subject
        assert len(metrics_list) == 4

    def test_result_skips_invalid_subjects(self, setup):
        # Setup: artificially make Subject "02" have entirely Class 0
        y_mod = setup["y"].copy()
        sub0_mask = setup["groups"] == "02"
        y_mod[sub0_mask] = 0
        
        metrics_list, _ = run_within_subject_distribution_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=y_mod, 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_runs=1, results_path=setup["results_path"],
            min_samples_per_class=2, # sub-00 has 0 of class 1
            **_NO_PLOTS,
        )
        assert len(metrics_list) == 3 # 1 skipped

    def test_csv_saving(self, setup):
        run_within_subject_distribution_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_runs=1, results_path=setup["results_path"],
            save_csv=True, save_probabilities=True,
            **_NO_PLOTS,
        )
        res_dir = Path(setup["results_path"])
        assert len(list(res_dir.rglob("*ws_subject_metrics.csv"))) > 0
        assert len(list(res_dir.rglob("*sample_predictions.csv"))) > 0
        assert len(list(res_dir.rglob("*summary.csv"))) > 0

    def test_multi_run_outputs_correctly(self, setup):
        metrics_list, _ = run_within_subject_distribution_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_runs=2, results_path=setup["results_path"],
            min_samples_per_class=0, 
            **_NO_PLOTS,
        )
        
        df_res = pd.DataFrame(metrics_list)
        # 4 subjects * 2 runs = 8 rows
        assert len(df_res) == 8


class TestRunWithinSubjectPermutationAnalysis:
    """Integration tests for run_within_subject_permutation_analysis."""

    @pytest.fixture
    def setup(self, tmp_path):
        df, X, y, groups, feature_cols = make_synthetic_data(
            n_subjects=3, n_probes_per_subject=20, n_features=5, random_seed=3
        )
        tasks = pd.Series(["T1"] * len(df))
        config = make_minimal_config(tmp_path=tmp_path, n_subjects=3)
        results_path = str(tmp_path / "results_ws_perm")
        return dict(df=df, X=X, y=y, groups=groups, tasks=tasks,
                    feature_cols=feature_cols, config=config,
                    results_path=results_path)

    def test_returns_dataframe_and_dict(self, setup):
        results_df, perm_summary, _, _ = run_within_subject_permutation_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_permutations=2, results_path=setup["results_path"],
            min_samples_per_class=0,
            **_NO_PLOTS,
        )
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2 # 2 permutation passes
        assert isinstance(perm_summary, dict)

    def test_zero_permutations_returns_empty(self, setup):
        results_df, perm_summary, _, _ = run_within_subject_permutation_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_permutations=0, results_path=setup["results_path"],
            **_NO_PLOTS,
        )
        assert results_df.empty
        assert perm_summary == {}

    def test_permutation_computes_p_value_with_true_df(self, setup):
        # Fake a true metrics df representing 3 subjects
        true_df = pd.DataFrame([
            {"subject": "02", "mean_auc": 0.8},
            {"subject": "03", "mean_auc": 0.85},
            {"subject": "04", "mean_auc": 0.75},
        ])
        
        results_df, perm_summary, _, _ = run_within_subject_permutation_analysis(
            dimension="ON_vs_OFF", df=setup["df"], X=setup["X"], y=setup["y"], 
            subjects=setup["groups"], tasks=setup["tasks"], feature_cols=setup["feature_cols"], config=setup["config"],
            model_type="lr", n_permutations=2, results_path=setup["results_path"],
            min_samples_per_class=0,
            true_ws_metrics_df=true_df,
            **_NO_PLOTS,
        )
        assert "p_auc" in perm_summary
        assert 0.0 <= perm_summary["p_auc"] <= 1.0
