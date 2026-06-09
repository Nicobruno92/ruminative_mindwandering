"""Shared utilities for behavioral classification pipeline.

Provides data loading, merging, and feature column resolution for the
behavior classification analyses (group classification, inclusion/exclusion
classification).

All paths are resolved relative to the current working directory (repo root),
so scripts must be run from the repository root.
"""

# === Imports ===
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# === Data Loading ===

def load_probe_data(config: dict) -> pd.DataFrame:
    """Load probe-level aggregated data (all probes, all subjects).

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Probe-level data with 2460 rows and all probe dimension columns,
        subject metadata, and group/inclusion_exclusion labels.
    """
    path = Path(config["data"]["probe_data_path"])
    return pd.read_csv(path)


def load_pca_data(config: dict) -> pd.DataFrame:
    """Load PCA results restricted to off-task probes (PC1, PC2, PC3 only).

    Off-task probes are those below the median onoff split; this file
    contains 888 rows. On-task probes will have NaN PCs after merging.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: subject, task, probe_number, PC1, PC2, PC3.
    """
    path = Path(config["data"]["pca_results_path"])
    df = pd.read_csv(path)
    return df[["subject", "task", "probe_number", "PC1", "PC2", "PC3"]]


def load_objective_markers(config: dict) -> pd.DataFrame:
    """Load SART objective performance markers per probe window.

    Returns the merge-key columns plus the marker columns listed under
    ``features.objective_marker_cols`` in the config.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: subject, task, probe_number, <objective_marker_cols>.
    """
    path = Path(config["data"]["objective_markers_path"])
    marker_cols: List[str] = config["features"]["objective_marker_cols"]
    return pd.read_csv(path)[["subject", "task", "probe_number"] + marker_cols]


def merge_all_features(config: dict) -> pd.DataFrame:
    """Load and left-join probe data, PCA components, and objective markers.

    PCA components are left-joined, so on-task probes (not in pca_results.csv)
    will have NaN for PC1/PC2/PC3. Objective markers span all 2460 probes.
    The returned DataFrame has the same number of rows as the probe data.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with no duplicate column names.
    """
    df = load_probe_data(config)
    merge_keys: List[str] = ["subject", "task", "probe_number"]

    if config["features"]["include_pca"]:
        pca = load_pca_data(config)
        df = df.merge(pca, on=merge_keys, how="left")

    if config["features"]["include_objective_markers"]:
        obj = load_objective_markers(config)
        df = df.merge(obj, on=merge_keys, how="left")

    return df


def get_feature_cols(config: dict) -> List[str]:
    """Return ordered list of feature column names based on config flags.

    Order: probe dimensions → objective markers → PCA components.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    List[str]
        Feature column names in the order they should be passed to classifiers.
    """
    cols: List[str] = list(config["features"]["probe_dimensions"])
    if config["features"]["include_objective_markers"]:
        cols += config["features"]["objective_marker_cols"]
    if config["features"]["include_pca"]:
        cols += ["PC1", "PC2", "PC3"]
    return cols


# === Feature Building ===

def build_subject_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features per subject (mean). Returns X, y, groups.

    Parameters
    ----------
    df : DataFrame with probe-level rows
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'group')
    positive_class : value encoded as 1 (e.g. 'Risk of Depression')

    Returns
    -------
    X : (n_subjects, n_features) float array
    y : (n_subjects,) int binary array
    groups : (n_subjects,) str array of subject IDs
    """
    agg = df.groupby("subject")[feature_cols].mean()
    y_raw = df.groupby("subject")[target_col].first()
    y = (y_raw == positive_class).astype(int).values
    groups = agg.index.astype(str).values
    return agg.values.astype(float), y, groups


def build_probe_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probe-level feature matrix. Label = subject's class (constant within subject).

    Parameters
    ----------
    df : DataFrame with probe-level rows
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'group')
    positive_class : value encoded as 1 (e.g. 'Risk of Depression')

    Returns
    -------
    X : (n_probes, n_features)
    y : (n_probes,) — same label for all probes of the same subject
    groups : (n_probes,) — subject IDs (LOSO fold key)
    """
    df = df.copy()
    subj_labels = df.groupby("subject")[target_col].first()
    df["_label"] = df["subject"].map(subj_labels)
    y = (df["_label"] == positive_class).astype(int).values
    groups = df["subject"].astype(str).values
    return df[feature_cols].values.astype(float), y, groups


def build_block_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features per subject × task block (mean). One row per block.

    Parameters
    ----------
    df : DataFrame already filtered to the relevant conditions (e.g. intervention only)
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'inclusion_exclusion')
    positive_class : value encoded as 1 (e.g. 'inclusion')

    Returns
    -------
    X : (n_blocks, n_features)
    y : (n_blocks,) binary
    groups : (n_blocks,) subject IDs (LOSO fold key)
    """
    agg = df.groupby(["subject", "task"])[feature_cols].mean()
    y_raw = df.groupby(["subject", "task"])[target_col].first()
    y = (y_raw == positive_class).astype(int).values
    groups = agg.index.get_level_values("subject").astype(str).values
    return agg.values.astype(float), y, groups


# === Model Building ===

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import joblib


def build_model(model_type: str, config: dict):
    """Instantiate a classifier with balanced class weights from config params.

    Parameters
    ----------
    model_type : 'lr' or 'rf'
    config : full config dict; reads from config['models']

    Returns
    -------
    sklearn classifier with class_weight='balanced'
    """
    rs = config["models"]["random_state"]
    if model_type == "lr":
        return LogisticRegression(
            C=config["models"]["lr_C"],
            penalty=config["models"]["lr_penalty"],
            solver="lbfgs",
            max_iter=config["models"]["lr_max_iter"],
            class_weight="balanced",
            random_state=rs,
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config["models"]["rf_n_estimators"],
            max_depth=config["models"]["rf_max_depth"],
            class_weight="balanced",
            n_jobs=1,
            random_state=rs,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}. Use 'lr' or 'rf'.")


# === LOSO Engine ===

def _extract_importances(model, n_features: int) -> np.ndarray:
    """Extract feature importances (RF) or absolute coefficients (LR) from fitted model."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return np.abs(model.coef_[0])
    return np.zeros(n_features)


def _run_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
) -> dict:
    """Process a single LOSO fold: impute → scale → fit → predict.

    Parameters
    ----------
    train_idx : indices for training samples
    test_idx : indices for test samples
    X : full feature matrix
    y : full label array
    groups : full group (subject) array
    model : unfitted sklearn estimator

    Returns
    -------
    dict with y_test, proba, importances, subject_metric
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    subject = groups[test_idx[0]]

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    m = clone(model)
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:, 1]
    importances = _extract_importances(m, X_train.shape[1])

    y_pred = (proba >= 0.5).astype(int)
    subj_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else np.nan

    return {
        "y_test": y_test,
        "proba": proba,
        "importances": importances,
        "subject_metric": {
            "subject": subject,
            "n_samples": len(y_test),
            "y_true_majority": int(y_test[0]) if len(np.unique(y_test)) == 1 else -1,
            "y_proba_mean": float(np.mean(proba)),
            "auc": subj_auc,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        },
    }


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    n_jobs: int = 1,
) -> dict:
    """Run LOSO cross-validation. Returns pooled metrics and per-subject details.

    Parameters
    ----------
    X : (n_samples, n_features) feature matrix
    y : (n_samples,) binary int label array
    groups : (n_samples,) subject ID string array
    model : sklearn-compatible classifier with class_weight='balanced'
    n_jobs : number of parallel fold workers (joblib)

    Returns
    -------
    dict with keys:
        auc : float — pooled ROC AUC
        balanced_accuracy : float — pooled balanced accuracy
        y_true : list[int]
        y_proba : list[float]
        subject_metrics : list[dict]
        feature_importances : np.ndarray (n_features,)
    """
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))

    fold_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_fold)(ti, tei, X, y, groups, model)
        for ti, tei in folds
    )

    y_true_all: List[int] = []
    y_proba_all: List[float] = []
    subject_metrics: List[dict] = []
    importances_list: List[np.ndarray] = []

    for fr in fold_results:
        y_true_all.extend(fr["y_test"].tolist())
        y_proba_all.extend(fr["proba"].tolist())
        importances_list.append(fr["importances"])
        subject_metrics.append(fr["subject_metric"])

    y_true_arr = np.array(y_true_all)
    y_proba_arr = np.array(y_proba_all)
    y_pred_arr = (y_proba_arr >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true_arr, y_proba_arr),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "y_true": y_true_all,
        "y_proba": y_proba_all,
        "subject_metrics": subject_metrics,
        "feature_importances": np.mean(importances_list, axis=0),
    }
