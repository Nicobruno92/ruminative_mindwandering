# LOSO Median-Position vs Decodability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone analysis script that produces two exploratory figures relating LOSO decodability (per-subject AUC) to (A) how consistent subjects' within-subject medians are across subjects, per dimension, and (B) how close each subject's own median sits to the scale midpoint (50).

**Architecture:** One standalone script `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py` with pure, independently-testable functions for data loading/shaping and two plotly figure builders, orchestrated by `main()`. Tests in `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py` cover the pure functions with synthetic data and the loaders against the real (already-existing) LOSO result files.

**Tech Stack:** Python (ML conda env), pandas, numpy, scipy.stats, plotly (graph_objects + subplots), pyyaml, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-10-loso-median-position-vs-decodability-design.md`

---

## Conventions for this plan

- Run all commands from the repo root: `/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering`
- Python interpreter: `/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python`
- Pytest (override the HTML/JUnit addopts from `loso_pipeline/pytest.ini` since they don't apply here):
  ```bash
  /network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
    mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
  ```

---

### Task 1: Script and test scaffolding

**Files:**
- Create: `mw_classification_pipeline/scripts/__init__.py`
- Create: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- Create: `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`

- [ ] **Step 1: Create empty package init for scripts**

```bash
touch mw_classification_pipeline/scripts/__init__.py
```

- [ ] **Step 2: Write the script skeleton with imports and constants**

Create `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`:

```python
#!/usr/bin/env python
"""
LOSO: AUC vs median position — exploratory analysis.

Two figures relating LOSO decodability (per-subject AUC) to within-subject
median ratings on the 5 phenomenology dimensions (onoff, valence, selfother,
time, confidence):

  Fig A (dimension_auc_vs_median_variability):
      One point per dimension. x = SD across subjects of each subject's
      within-subject median rating. y = mean LOSO AUC for that dimension.

  Fig B (auc_vs_median_distance_from_50_faceted):
      One panel per dimension, one point per subject. x = |subject's median
      rating - 50| (distance from scale midpoint). y = subject's LOSO AUC.

Status: EXPLORATORY (n=5 dimensions for Fig A; no confirmatory claims).

Usage (from project root):
    /path/to/miniforge3/envs/ML/bin/python \
        mw_classification_pipeline/scripts/plot_auc_vs_median_position.py
"""

# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plotting_utils import COLORS  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOSO_RESULTS_ROOT = (
    PROJECT_ROOT / "mw_classification_pipeline" / "results" / "MW_Classification" / "LOSO"
)
PROBE_DATA_PATH = (
    PROJECT_ROOT / "results" / "Behavior" / "probe_data" / "probe_level_aggregated_data.csv"
)
OUTPUT_DIR = LOSO_RESULTS_ROOT / "median_position_analysis"

SCALE_MIDPOINT = 50.0
CHANCE_AUC = 0.5
MIN_POINTS_FOR_FIT = 3

# (contrast directory name, display label, marker color) — color scheme
# matches scripts/generate_combined_classification_figure.py
DIMENSIONS: List[Dict[str, str]] = [
    {"contrast": "ON_vs_OFF_within_median", "label": "On/Off-Task", "color": "#DE237B"},
    {"contrast": "valence_within_median", "label": "Valence", "color": "#7B4FBA"},
    {"contrast": "selfother_within_median", "label": "Self/Other", "color": "#E67E22"},
    {"contrast": "confidence_within_median", "label": "Confidence", "color": "#27AE60"},
    {"contrast": "time_within_median", "label": "Time", "color": "#2980B9"},
]
```

- [ ] **Step 3: Write the test file skeleton**

Create `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`:

```python
"""Tests for scripts/plot_auc_vs_median_position.py. Synthetic data + real LOSO fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_auc_vs_median_position import LOSO_RESULTS_ROOT
```

- [ ] **Step 4: Run the test file to confirm it imports cleanly**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: collects 0 tests, no import errors (`no tests ran`).

- [ ] **Step 5: Commit scaffolding**

```bash
git add mw_classification_pipeline/scripts/__init__.py \
        mw_classification_pipeline/scripts/plot_auc_vs_median_position.py \
        mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py
git commit -m "exp: scaffold LOSO median-position vs decodability analysis script"
```

---

### Task 2: Core data-shaping functions (compute_pearson_fit, compute_subject_medians, build_fig_a_row, build_fig_b_rows)

**Files:**
- Modify: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- Modify: `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`

- [ ] **Step 1: Write failing tests for the four pure functions**

Append to `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`:

```python
from scripts.plot_auc_vs_median_position import (
    compute_pearson_fit,
    compute_subject_medians,
    build_fig_a_row,
    build_fig_b_rows,
)


def test_compute_pearson_fit_perfect_positive_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = compute_pearson_fit(x, y)
    assert result["r"] == pytest.approx(1.0)
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert result["n"] == 5


def test_compute_pearson_fit_returns_nan_below_three_points():
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    result = compute_pearson_fit(x, y)
    assert np.isnan(result["r"])
    assert np.isnan(result["slope"])
    assert result["n"] == 2


def test_compute_subject_medians_filters_and_computes_median():
    probe_df = pd.DataFrame({
        "subject": ["02", "02", "02", "03", "03", "04", "04"],
        "onoff": [10.0, 50.0, 90.0, 40.0, 60.0, 0.0, 100.0],
    })
    result = compute_subject_medians(probe_df, "onoff", subjects_final=["02", "03"])
    result = result.set_index("subject")
    assert result.loc["02", "subject_median"] == pytest.approx(50.0)
    assert result.loc["03", "subject_median"] == pytest.approx(50.0)
    assert "04" not in result.index


def test_build_fig_a_row_computes_sd_and_mean():
    subject_medians = pd.DataFrame({
        "subject": ["02", "03", "04"],
        "subject_median": [40.0, 50.0, 60.0],
    })
    subject_aucs = pd.DataFrame({
        "subject": ["02", "03", "04"],
        "auc": [0.6, 0.7, 0.8],
    })
    row = build_fig_a_row("Test Dim", subject_medians, subject_aucs)
    assert row["dimension"] == "Test Dim"
    assert row["median_sd"] == pytest.approx(10.0)
    assert row["mean_auc"] == pytest.approx(0.7)
    assert row["n_subjects"] == 3


def test_build_fig_b_rows_computes_distance_and_merges():
    subject_medians = pd.DataFrame({
        "subject": ["02", "03"],
        "subject_median": [50.0, 80.0],
    })
    subject_aucs = pd.DataFrame({
        "subject": ["02", "03"],
        "auc": [0.65, 0.55],
    })
    result = build_fig_b_rows("Test Dim", subject_medians, subject_aucs).set_index("subject")
    assert result.loc["02", "dist_from_50"] == pytest.approx(0.0)
    assert result.loc["03", "dist_from_50"] == pytest.approx(30.0)
    assert (result["dimension"] == "Test Dim").all()
    assert list(result.columns) == ["dimension", "subject_median", "dist_from_50", "auc"]
```

Note: the last assertion's column order assumes `build_fig_b_rows` returns
`["dimension", "subject", "subject_median", "dist_from_50", "auc"]` and we
then `set_index("subject")`, leaving `["dimension", "subject_median",
"dist_from_50", "auc"]`.

- [ ] **Step 2: Run tests to verify they fail with ImportError / NameError**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: FAIL — `ImportError: cannot import name 'compute_pearson_fit'`.

- [ ] **Step 3: Implement the four functions**

Append to `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
(after the configuration block):

```python
# =============================================================================
# Core data-shaping functions
# =============================================================================

def compute_pearson_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation and OLS line fit between two 1-D arrays.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length arrays of paired observations.

    Returns
    -------
    Dict[str, float]
        Keys 'r', 'p', 'slope', 'intercept', 'n'. If fewer than
        `MIN_POINTS_FOR_FIT` finite pairs are available, 'r', 'p', 'slope'
        and 'intercept' are NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())

    if n_valid < MIN_POINTS_FOR_FIT:
        return {"r": np.nan, "p": np.nan, "slope": np.nan, "intercept": np.nan, "n": n_valid}

    r_val, p_val = stats.pearsonr(x[valid], y[valid])
    slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])
    return {
        "r": float(r_val),
        "p": float(p_val),
        "slope": float(slope),
        "intercept": float(intercept),
        "n": n_valid,
    }


def compute_subject_medians(
    probe_df: pd.DataFrame, label_source: str, subjects_final: List[str]
) -> pd.DataFrame:
    """
    Per-subject median rating for one dimension, restricted to a subject set.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Probe-level data with a 'subject' column (zero-padded strings) and
        a `label_source` column.
    label_source : str
        Name of the dimension column (e.g. 'onoff', 'valence').
    subjects_final : List[str]
        Zero-padded subject IDs to keep — the LOSO subject set for this
        dimension.

    Returns
    -------
    pd.DataFrame
        Columns: 'subject', 'subject_median'.
    """
    medians = (
        probe_df.groupby("subject")[label_source]
        .median()
        .reset_index()
        .rename(columns={label_source: "subject_median"})
    )
    return medians[medians["subject"].isin(subjects_final)].reset_index(drop=True)


def build_fig_a_row(
    dimension_label: str, subject_medians: pd.DataFrame, subject_aucs: pd.DataFrame
) -> Dict[str, object]:
    """
    One Fig A summary row: across-subject SD of medians vs mean LOSO AUC.

    Parameters
    ----------
    dimension_label : str
        Display name of the dimension (e.g. 'On/Off-Task').
    subject_medians : pd.DataFrame
        Output of `compute_subject_medians` — columns 'subject', 'subject_median'.
    subject_aucs : pd.DataFrame
        Output of `load_subject_aucs` — columns 'subject', 'auc'.

    Returns
    -------
    Dict[str, object]
        Keys: 'dimension', 'median_sd', 'mean_auc', 'n_subjects'.
    """
    return {
        "dimension": dimension_label,
        "median_sd": float(subject_medians["subject_median"].std()),
        "mean_auc": float(subject_aucs["auc"].mean()),
        "n_subjects": int(len(subject_aucs)),
    }


def build_fig_b_rows(
    dimension_label: str, subject_medians: pd.DataFrame, subject_aucs: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-subject Fig B rows: distance of the subject's median from the scale
    midpoint vs that subject's LOSO AUC.

    Parameters
    ----------
    dimension_label : str
        Display name of the dimension (e.g. 'On/Off-Task').
    subject_medians : pd.DataFrame
        Output of `compute_subject_medians`.
    subject_aucs : pd.DataFrame
        Output of `load_subject_aucs`.

    Returns
    -------
    pd.DataFrame
        Columns: 'dimension', 'subject', 'subject_median', 'dist_from_50', 'auc'.
    """
    merged = pd.merge(subject_medians, subject_aucs, on="subject", how="inner")
    merged["dist_from_50"] = (merged["subject_median"] - SCALE_MIDPOINT).abs()
    merged["dimension"] = dimension_label
    return merged[["dimension", "subject", "subject_median", "dist_from_50", "auc"]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: PASS — 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/scripts/plot_auc_vs_median_position.py \
        mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py
git commit -m "exp: add core data-shaping functions for median-position analysis"
```

---

### Task 3: Data loaders against real LOSO results (load_dimension_metadata, load_subject_aucs)

**Files:**
- Modify: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- Modify: `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`

- [ ] **Step 1: Write failing tests against the real ON_vs_OFF LOSO result**

Append to `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`:

```python
from scripts.plot_auc_vs_median_position import (
    load_dimension_metadata,
    load_subject_aucs,
)


ON_OFF_RF_DIR = (
    LOSO_RESULTS_ROOT / "ON_vs_OFF_within_median" / "all" / "rf"
)


def test_load_dimension_metadata_on_off():
    subjects_final, label_source = load_dimension_metadata(ON_OFF_RF_DIR)
    assert label_source == "onoff"
    assert len(subjects_final) == 29
    assert "03" in subjects_final
    assert "16" not in subjects_final  # excluded_no_data_or_all_neutral
    assert all(len(s) == 2 for s in subjects_final)


def test_load_subject_aucs_on_off():
    subjects_final, _ = load_dimension_metadata(ON_OFF_RF_DIR)
    aucs = load_subject_aucs(ON_OFF_RF_DIR, subjects_final)
    assert len(aucs) == 29
    assert set(aucs["subject"]) == set(subjects_final)
    assert aucs["auc"].mean() == pytest.approx(0.628, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail with ImportError**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: FAIL — `ImportError: cannot import name 'load_dimension_metadata'`.

- [ ] **Step 3: Implement the two loader functions**

Append to `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`:

```python
# =============================================================================
# Data loaders
# =============================================================================

def load_dimension_metadata(rf_dir: Path) -> Tuple[List[str], str]:
    """
    Read the LOSO `used_config.yaml` for one dimension.

    Parameters
    ----------
    rf_dir : Path
        Directory containing `used_config.yaml` and the subject-metrics CSV
        (e.g. `.../LOSO/ON_vs_OFF_within_median/all/rf`).

    Returns
    -------
    Tuple[List[str], str]
        (subjects_final as zero-padded strings, label_source column name).
    """
    config = yaml.safe_load((rf_dir / "used_config.yaml").read_text())
    contrast = config["contrast"]
    subjects_final = [str(s).zfill(2) for s in config["_data_provenance"]["subjects_final"]]
    label_source = config["label_contrasts"][contrast]["label_source"]
    return subjects_final, label_source


def load_subject_aucs(rf_dir: Path, subjects_final: List[str]) -> pd.DataFrame:
    """
    Load per-subject LOSO AUC, restricted to `subjects_final`.

    Parameters
    ----------
    rf_dir : Path
        Directory containing `rf_loso_100runs_loso_subject_metrics.csv`.
    subjects_final : List[str]
        Zero-padded subject IDs to keep.

    Returns
    -------
    pd.DataFrame
        Columns: 'subject' (zero-padded string), 'auc'.
    """
    df = pd.read_csv(rf_dir / "rf_loso_100runs_loso_subject_metrics.csv", dtype={"subject": str})
    df["subject"] = df["subject"].str.zfill(2)
    df = df[df["subject"].isin(subjects_final)][["subject", "auc"]]
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: PASS — 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/scripts/plot_auc_vs_median_position.py \
        mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py
git commit -m "exp: add LOSO result loaders for median-position analysis"
```

---

### Task 4: Fig A — dimension-level scatter

**Files:**
- Modify: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- Modify: `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`

- [ ] **Step 1: Write a failing smoke test**

Append to `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`:

```python
from scripts.plot_auc_vs_median_position import plot_fig_a


def test_plot_fig_a_creates_output_files(tmp_path):
    fig_a_df = pd.DataFrame({
        "dimension": ["On/Off-Task", "Valence"],
        "median_sd": [10.0, 20.0],
        "mean_auc": [0.628, 0.571],
        "n_subjects": [29, 27],
    })
    plot_fig_a(fig_a_df, tmp_path)
    assert (tmp_path / "dimension_auc_vs_median_variability.png").exists()
    assert (tmp_path / "dimension_auc_vs_median_variability.pdf").exists()
    assert (tmp_path / "dimension_auc_vs_median_variability.html").exists()
```

- [ ] **Step 2: Run test to verify it fails with ImportError**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: FAIL — `ImportError: cannot import name 'plot_fig_a'`.

- [ ] **Step 3: Implement `plot_fig_a`**

Append to `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`:

```python
# =============================================================================
# Fig A: dimension-level scatter
# =============================================================================

def plot_fig_a(fig_a_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter of dimension-level median variability vs mean LOSO AUC.

    One point per dimension (n=5 in the full run). Exploratory — no
    correlation statistic is computed given the small n.

    Parameters
    ----------
    fig_a_df : pd.DataFrame
        Columns: 'dimension', 'median_sd', 'mean_auc', 'n_subjects'.
    output_dir : Path
        Directory to write `dimension_auc_vs_median_variability.{png,pdf,html}`.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fig_a_df["median_sd"],
        y=fig_a_df["mean_auc"],
        mode="markers+text",
        text=fig_a_df["dimension"],
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(color=COLORS[0], size=14, opacity=0.85, line=dict(color="white", width=1)),
        hovertemplate="%{text}<br>Median SD: %{x:.2f}<br>Mean AUC: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=CHANCE_AUC, line_dash="dash", line_color="gray", opacity=0.5,
        annotation_text="Chance (0.5)", annotation_position="bottom right",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=(
                "<b>LOSO: Dimension Decodability vs Across-Subject Median Variability</b>"
                f"<br><sup>Exploratory — n = {len(fig_a_df)} dimensions, "
                "descriptive only (no correlation statistic)</sup>"
            ),
            font=dict(size=15),
        ),
        xaxis_title="SD across subjects of within-subject median rating",
        yaxis_title="Mean LOSO AUC",
        width=750,
        height=600,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dimension_auc_vs_median_variability"
    fig.write_image(f"{out_path}.png", scale=2)
    fig.write_image(f"{out_path}.pdf")
    fig.write_html(f"{out_path}.html")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: PASS — 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/scripts/plot_auc_vs_median_position.py \
        mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py
git commit -m "exp: add Fig A (dimension-level median variability vs AUC) plot"
```

---

### Task 5: Fig B — subject-level faceted scatter

**Files:**
- Modify: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`
- Modify: `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`

- [ ] **Step 1: Write a failing smoke test**

Append to `mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py`:

```python
from scripts.plot_auc_vs_median_position import plot_fig_b


def test_plot_fig_b_creates_output_files(tmp_path):
    rng = np.random.default_rng(42)
    dimension_order = ["On/Off-Task", "Valence", "Self/Other", "Confidence", "Time"]
    dimension_colors = {d: "#000000" for d in dimension_order}

    rows = []
    for dim in dimension_order:
        for i in range(5):
            rows.append({
                "dimension": dim,
                "subject": f"{i + 2:02d}",
                "subject_median": 50.0,
                "dist_from_50": float(rng.uniform(0, 30)),
                "auc": float(rng.uniform(0.4, 0.8)),
            })
    fig_b_df = pd.DataFrame(rows)

    plot_fig_b(fig_b_df, dimension_colors, dimension_order, tmp_path)

    assert (tmp_path / "auc_vs_median_distance_from_50_faceted.png").exists()
    assert (tmp_path / "auc_vs_median_distance_from_50_faceted.pdf").exists()
    assert (tmp_path / "auc_vs_median_distance_from_50_faceted.html").exists()
```

- [ ] **Step 2: Run test to verify it fails with ImportError**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: FAIL — `ImportError: cannot import name 'plot_fig_b'`.

- [ ] **Step 3: Implement `plot_fig_b`**

Append to `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`:

```python
# =============================================================================
# Fig B: subject-level faceted scatter
# =============================================================================

def _subplot_domain_ref(index: int, axis: str) -> str:
    """Return the plotly domain ref ('x domain', 'x2 domain', ...) for subplot `index` (0-based)."""
    suffix = "" if index == 0 else str(index + 1)
    return f"{axis}{suffix} domain"


def plot_fig_b(
    fig_b_df: pd.DataFrame,
    dimension_colors: Dict[str, str],
    dimension_order: List[str],
    output_dir: Path,
) -> None:
    """
    Faceted scatter of |subject median - 50| vs subject LOSO AUC, one panel
    per dimension, in a 2x3 grid (5 panels + 1 empty cell).

    Parameters
    ----------
    fig_b_df : pd.DataFrame
        Columns: 'dimension', 'subject', 'subject_median', 'dist_from_50', 'auc'.
    dimension_colors : Dict[str, str]
        Maps dimension display label to a hex marker color.
    dimension_order : List[str]
        Display labels in the order panels should appear (length 5).
    output_dir : Path
        Directory to write `auc_vs_median_distance_from_50_faceted.{png,pdf,html}`.
    """
    fig = make_subplots(rows=2, cols=3, subplot_titles=dimension_order)

    y_min = max(0.0, float(fig_b_df["auc"].min()) - 0.05)
    y_max = min(1.0, float(fig_b_df["auc"].max()) + 0.05)

    for i, dimension in enumerate(dimension_order):
        row = i // 3 + 1
        col = i % 3 + 1
        sub = fig_b_df[fig_b_df["dimension"] == dimension]
        x = sub["dist_from_50"].to_numpy(dtype=float)
        y = sub["auc"].to_numpy(dtype=float)
        fit = compute_pearson_fit(x, y)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text",
            text=sub["subject"], textposition="top center", textfont=dict(size=8),
            marker=dict(
                color=dimension_colors[dimension], size=8, opacity=0.85,
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            hovertemplate="Subject: %{text}<br>|median-50|: %{x:.1f}<br>AUC: %{y:.3f}<extra></extra>",
        ), row=row, col=col)

        if np.isfinite(fit["slope"]):
            x_line = np.array([x.min(), x.max()])
            y_line = fit["slope"] * x_line + fit["intercept"]
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="black", width=2, dash="dash"), showlegend=False,
            ), row=row, col=col)

        fig.add_hline(y=CHANCE_AUC, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col)

        annotation_text = (
            f"r = {fit['r']:.2f}, p = {fit['p']:.3f}" if np.isfinite(fit["r"]) else "n < 3"
        )
        fig.add_annotation(
            text=annotation_text,
            xref=_subplot_domain_ref(i, "x"), yref=_subplot_domain_ref(i, "y"),
            x=0.98, y=0.02, xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=10),
        )

    fig.update_yaxes(range=[y_min, y_max], title_text="LOSO AUC")
    fig.update_xaxes(title_text="|subject median - 50|")

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>LOSO: Subject AUC vs Distance of Median from Scale Midpoint (50)</b>",
            font=dict(size=15),
        ),
        height=700,
        width=1100,
        showlegend=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "auc_vs_median_distance_from_50_faceted"
    fig.write_image(f"{out_path}.png", scale=2)
    fig.write_image(f"{out_path}.pdf")
    fig.write_html(f"{out_path}.html")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: PASS — 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/scripts/plot_auc_vs_median_position.py \
        mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py
git commit -m "exp: add Fig B (subject AUC vs median distance from 50, faceted) plot"
```

---

### Task 6: Orchestration (`main`), end-to-end run, and scientific review

**Files:**
- Modify: `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`

- [ ] **Step 1: Implement `main()`**

Append to `mw_classification_pipeline/scripts/plot_auc_vs_median_position.py`:

```python
# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build Fig A and Fig B and write data CSVs + provenance to OUTPUT_DIR."""
    probe_df = pd.read_csv(PROBE_DATA_PATH, dtype={"subject": str})
    probe_df["subject"] = probe_df["subject"].astype(str).str.zfill(2)

    fig_a_rows: List[Dict[str, object]] = []
    fig_b_frames: List[pd.DataFrame] = []
    dimension_colors: Dict[str, str] = {}
    dimension_order: List[str] = []
    provenance: Dict[str, Dict[str, object]] = {}

    for dim in DIMENSIONS:
        rf_dir = LOSO_RESULTS_ROOT / dim["contrast"] / "all" / "rf"
        subjects_final, label_source = load_dimension_metadata(rf_dir)
        subject_aucs = load_subject_aucs(rf_dir, subjects_final)
        subject_medians = compute_subject_medians(probe_df, label_source, subjects_final)

        fig_a_rows.append(build_fig_a_row(dim["label"], subject_medians, subject_aucs))
        fig_b_frames.append(build_fig_b_rows(dim["label"], subject_medians, subject_aucs))

        dimension_colors[dim["label"]] = dim["color"]
        dimension_order.append(dim["label"])
        provenance[dim["label"]] = {
            "contrast": dim["contrast"],
            "label_source": label_source,
            "n_subjects": int(len(subject_aucs)),
            "subjects_final": subjects_final,
        }

    fig_a_df = pd.DataFrame(fig_a_rows)
    fig_b_df = pd.concat(fig_b_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_a_df.to_csv(OUTPUT_DIR / "dimension_auc_vs_median_variability_data.csv", index=False)
    fig_b_df.to_csv(OUTPUT_DIR / "auc_vs_median_distance_from_50_data.csv", index=False)

    plot_fig_a(fig_a_df, OUTPUT_DIR)
    plot_fig_b(fig_b_df, dimension_colors, dimension_order, OUTPUT_DIR)

    used_config = {
        "analysis": "loso_median_position_vs_decodability",
        "status": "exploratory",
        "data_sources": {
            "probe_data": str(PROBE_DATA_PATH.relative_to(PROJECT_ROOT)),
            "loso_results_root": str(LOSO_RESULTS_ROOT.relative_to(PROJECT_ROOT)),
        },
        "scale_midpoint": SCALE_MIDPOINT,
        "dimensions": provenance,
    }
    (OUTPUT_DIR / "used_config.yaml").write_text(yaml.safe_dump(used_config, sort_keys=False))

    print(f"Saved Fig A + Fig B + data CSVs + used_config.yaml -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite once more**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python -m pytest \
  mw_classification_pipeline/tests/test_plot_auc_vs_median_position.py -v -o addopts=""
```
Expected: PASS — 9 tests pass (main is not unit-tested directly; it's an
integration of already-tested pieces, exercised end-to-end in Step 3).

- [ ] **Step 3: Run the script end-to-end**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python \
  mw_classification_pipeline/scripts/plot_auc_vs_median_position.py
```
Expected: prints `Saved Fig A + Fig B + data CSVs + used_config.yaml -> .../median_position_analysis`,
and creates in `mw_classification_pipeline/results/MW_Classification/LOSO/median_position_analysis/`:
- `dimension_auc_vs_median_variability.{png,pdf,html}`
- `dimension_auc_vs_median_variability_data.csv` (5 rows)
- `auc_vs_median_distance_from_50_faceted.{png,pdf,html}`
- `auc_vs_median_distance_from_50_data.csv` (~140 rows: 29+27+32+28+24)
- `used_config.yaml`

- [ ] **Step 4: Scientific self-evaluation of the outputs**

Read both PNGs (`dimension_auc_vs_median_variability.png` and
`auc_vs_median_distance_from_50_faceted.png`) and the two data CSVs. Check:
- Fig A: 5 labeled points (On/Off-Task, Valence, Self/Other, Confidence,
  Time), y-values close to the known LOSO means (~0.628, 0.571, 0.515,
  0.586, 0.535 — see Task 3's loaded value for onoff as a reference).
- Fig B: 5 panels, each with the expected subject count (29/27/32/28/24),
  x-axis values in [0, 50], y-axis values mostly in [0.3, 0.85], chance
  line visible at 0.5.
- `used_config.yaml` correctly lists `subjects_final` per dimension and
  marks `status: exploratory`.

If anything looks implausible (e.g. all dist_from_50 == 0, or AUCs outside
[0,1]), stop and investigate before proceeding — do not adjust the analysis
to make it "look better".

- [ ] **Step 5: Commit the script (results/ stays untracked — gitignored)**

```bash
git add mw_classification_pipeline/scripts/plot_auc_vs_median_position.py
git commit -m "exp: orchestrate LOSO median-position vs decodability figures"
```

---

## Summary of what this produces

After Task 6, running
`mw_classification_pipeline/scripts/plot_auc_vs_median_position.py` produces
two new exploratory figures answering:

- **Fig A**: across the 5 MW dimensions, is group-level LOSO decodability
  related to how similar subjects' within-subject medians are to each other?
- **Fig B**: within each dimension, is a subject's LOSO AUC related to how
  close their own median sits to the 0-100 scale midpoint?

Both are clearly marked exploratory (n=5 for Fig A; per-panel Pearson r/p for
Fig B without multiple-comparisons correction across the 5 panels — note this
in any write-up).
