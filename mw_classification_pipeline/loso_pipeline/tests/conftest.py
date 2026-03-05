"""
Shared fixtures for the LOSO classification pipeline tests.

All tests use synthetic data only — no real EEG data or files needed.
Fixtures are deterministic and fast (<1s each).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Allow imports from the pipeline root (mw_classification_pipeline/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def make_synthetic_data(
    n_subjects: int = 6,
    n_probes_per_subject: int = 30,
    n_features: int = 20,
    random_seed: int = 42,
    feature_prefix: str = "feat",
) -> tuple:
    """
    Generate a deterministic, synthetic MW dataset.

    Creates a probe-level DataFrame with binary on/off-task labels and
    normally distributed features. Feature 0 carries a weak class signal.

    Parameters
    ----------
    n_subjects : int
        Number of subjects. Subject IDs are zero-padded strings ('02' onward).
    n_probes_per_subject : int
        Probes per subject.
    n_features : int
        Number of feature columns.
    random_seed : int
        Seed for reproducibility.
    feature_prefix : str
        Prefix for feature column names.

    Returns
    -------
    tuple
        (df, X, y, groups, feature_cols)
    """
    rng = np.random.default_rng(random_seed)
    subjects = [f"{i + 2:02d}" for i in range(n_subjects)]
    tasks = ["Sart1", "Sart2"]
    feature_cols = [f"{feature_prefix}_{j}" for j in range(n_features)]

    rows = []
    for sub in subjects:
        for probe_idx in range(n_probes_per_subject):
            task = tasks[probe_idx % len(tasks)]
            onoff = rng.integers(0, 101)
            target = 1 if onoff > 50 else 0
            feat_vals = rng.standard_normal(n_features)
            # Weak signal on the first feature to make classification non-trivial
            feat_vals[0] += target * 0.5
            row = {
                "subject": sub,
                "task": task,
                "probe_number": probe_idx,
                "onoff": int(onoff),
                "target": target,
            }
            for j, col in enumerate(feature_cols):
                row[col] = float(feat_vals[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    X = df[feature_cols].copy()
    y = df["target"].copy()
    groups = df["subject"].copy()
    return df, X, y, groups, feature_cols


def make_minimal_config(tmp_path=None, n_subjects: int = 6) -> dict:
    """
    Return a minimal pipeline config dict suitable for unit tests.

    Uses tmp_path for all I/O. Plots and SHAP are disabled to keep tests fast.

    Parameters
    ----------
    tmp_path : Path, optional
        Directory for results output.
    n_subjects : int
        Number of subjects in the synthetic dataset.

    Returns
    -------
    dict
    """
    results_root = str(tmp_path / "results") if tmp_path else "/tmp/mw_test_results"
    subjects = [f"{i + 2:02d}" for i in range(n_subjects)]

    return {
        "data_paths": {
            "features_root": "/nonexistent/path",
            "results_root": results_root,
        },
        "subjects": subjects,
        "tasks": ["Sart1", "Sart2"],
        "epoch_types": ["state"],
        "data_format": "per_channel",
        "label_source": "onoff",
        "label_contrasts": {
            "ON_vs_OFF": {
                "label_source": "onoff",
                "threshold": 50,
                "positive_above": True,
            },
            "ON_vs_OFF_extreme": {
                "label_source": "onoff",
                "threshold_low": 30,
                "threshold_high": 70,
            },
        },
        "contrast": "ON_vs_OFF",
        "model_type": "lr",
        "n_runs": 1,
        "permutation_runs": 0,
        "random_seed": 42,
        "min_samples": 5,
        "min_minority_ratio": 0.1,
        "k": 5,
        "feature_selection_method": "f_classif",
        "scaler": "standard",
        "scale_by_participant": "none",
        "permutation_scope": "global",
        "use_smote": False,
        "oversampling_method": "SMOTE",
        "oversampling_scope": "global",
        "use_pca": False,
        "pca_n_components": 5,
        "pca_type": "standard",
        "pca_kernel": "rbf",
        "scoring_metrics": ["auc", "balanced_accuracy", "auprc", "mcc"],
        "save_csv": True,
        "save_probabilities": True,
        "save_pickle": False,
        "save_plots": False,
        "save_shap": False,
        "verbose": False,
        "top_n_features_plot": 5,
        "plot_style": "seaborn",
        "current_model": "ON_vs_OFF",
        "results_folder_pattern": "",
    }


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_data():
    """Standard synthetic dataset: 6 subjects × 30 probes × 20 features."""
    return make_synthetic_data(n_subjects=6, n_probes_per_subject=30, n_features=20)


@pytest.fixture
def small_synthetic_data():
    """Minimal synthetic dataset for fast integration tests: 5 subjects × 20 probes × 10 features."""
    return make_synthetic_data(n_subjects=5, n_probes_per_subject=20, n_features=10, random_seed=99)


@pytest.fixture
def basic_config(tmp_path):
    """Minimal config dict with tmp_path for output."""
    return make_minimal_config(tmp_path=tmp_path, n_subjects=6)


@pytest.fixture
def small_config(tmp_path):
    """Minimal config for 5-subject synthetic dataset."""
    return make_minimal_config(tmp_path=tmp_path, n_subjects=5)
