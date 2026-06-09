# Spatial Decoding Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-electrode searchlight decoding mode to the LOSO and Within-Subject MW classification pipelines that produces topomap-ready metrics (AUC + permutation significance per channel) across all 5 probe dimensions.

**Architecture:** Option C — a shared searchlight core in `utils/spatial_decoding_utils.py` (channel parsing, channel loop, FDR, MNE montage, topomaps) plus one thin driver per pipeline. Each driver provides a `channel_eval` adapter that calls the *existing* analysis + permutation engine on the channel-restricted feature matrix and returns a normalized result dict. The per-electrode model is therefore provably identical to the main pipeline, restricted to one channel's columns.

**Tech Stack:** Python, scikit-learn (existing engine), pandas, numpy, MNE-Python (topomaps, `standard_1020` montage), pytest, SLURM. Conda env `ML` for runs, test runner per project memory.

**Spec:** `docs/superpowers/specs/2026-06-08-spatial-decoding-design.md`

> **REVISION (2026-06-09, during execution):** an implementation smoke test showed that
> per-channel permutation + Benjamini-Hochberg FDR (Tasks 2/4/6/7 below) is both
> statistically underpowered (min adjusted p ≈ 0.13 across 64 channels at 500 perms) and
> computationally prohibitive. The correction was changed to a **max-statistic permutation
> test (FWER)**, matching the Section-2 CBPT logic. This re-shaped the driver/merge/SLURM
> layers: the permutation axis is now **(dimension × permutation index)** — one job draws a
> single within-subject shuffle and scores all channels, so the per-permutation
> max-over-channels AUC is a valid family-wise null draw. The shared-core tasks (1, 3, 5, 8)
> and the per-channel true-run searchlight are unchanged and still valid. See the updated
> spec §6 and §8 for the authoritative design. The FDR helper (`fdr_correct`) remains in the
> code as an optional secondary correction.

**Key conventions (from project memory):**
- Permutation p-value uses the **+1 convention**: `p = (1 + #{null >= true}) / (1 + n_perm)`.
- Subject IDs are zero-padded strings; tasks are `Sart1..Sart4`.
- No `try/except` in scientific code; explicit `random_state` from config; no hardcoded paths/params.

---

## File Structure

**Created:**
- `mw_classification_pipeline/utils/spatial_decoding_utils.py` — shared core (pure-ish: parsing, FDR, montage, topomaps, channel-loop orchestration).
- `mw_classification_pipeline/tests/__init__.py` — make tests a package (if absent).
- `mw_classification_pipeline/tests/conftest.py` — synthetic per-channel data fixtures.
- `mw_classification_pipeline/tests/test_spatial_decoding.py` — unit + smoke tests for the core.
- `mw_classification_pipeline/loso_pipeline/spatial_decoding/run_loso_spatial_decoding.py` — LOSO driver + adapter.
- `mw_classification_pipeline/loso_pipeline/spatial_decoding/config.yaml` — LOSO spatial config.
- `mw_classification_pipeline/loso_pipeline/spatial_decoding/merge_spatial_results.py` — aggregate per-(dim,channel) jobs.
- `mw_classification_pipeline/loso_pipeline/spatial_decoding/run_spatial_slurm.sh` — SLURM array.
- `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/run_within_spatial_decoding.py` — WS driver + adapter.
- `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/config.yaml` — WS spatial config.
- `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/merge_spatial_results.py`
- `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/run_spatial_slurm.sh`

**Modified:** none of the existing pipeline files. The extension is additive.

### Normalized adapter contract (the interface between drivers and core)

Every `channel_eval(X_ch, channel, config, ...)` adapter returns:

```python
{
    "channel": str,            # electrode name, e.g. "Pz"
    "n_features": int,         # markers available for this channel
    "mean_auc": float,         # true mean AUC across runs (group-level for WS)
    "std_auc": float,          # std across runs
    "null_aucs": list[float],  # permutation null distribution of the same statistic
    "subject_auc": dict | None # WS only: {subject_id: mean_auc}; None for LOSO
}
```

The core computes the +1 p-value from `mean_auc` vs `null_aucs`, applies FDR across channels, and never needs to know whether the run was LOSO or WS.

---

## Task 1: Channel parsing and column selection (pure functions)

**Files:**
- Create: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Create: `mw_classification_pipeline/tests/__init__.py` (empty)
- Create: `mw_classification_pipeline/tests/conftest.py`
- Create: `mw_classification_pipeline/tests/test_spatial_decoding.py`

- [ ] **Step 1: Write the conftest synthetic per-channel data fixture**

`mw_classification_pipeline/tests/conftest.py`:

```python
"""Shared fixtures for spatial decoding tests. Synthetic data only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Allow `from utils.spatial_decoding_utils import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_per_channel_X(channels, markers, n_samples=40, random_seed=42):
    """Build a synthetic per-channel feature matrix with `{channel}_{marker}` columns."""
    rng = np.random.default_rng(random_seed)
    cols = [f"{ch}_{mk}" for ch in channels for mk in markers]
    data = rng.standard_normal((n_samples, len(cols)))
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def per_channel_X():
    # Includes P1 and P10 to guard against substring matching bugs.
    return make_per_channel_X(
        channels=["Fz", "Pz", "P1", "P10"],
        markers=["psd_bands_delta_mean", "P3b", "slowwaves_density"],
    )
```

- [ ] **Step 2: Write failing tests for parsing + selection**

`mw_classification_pipeline/tests/test_spatial_decoding.py`:

```python
"""Unit and smoke tests for utils/spatial_decoding_utils.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    select_channel_columns,
)


def test_parse_channels_recovers_ordered_unique_set(per_channel_X):
    channels = parse_channels_from_columns(per_channel_X.columns.tolist())
    assert channels == ["Fz", "P1", "P10", "Pz"]  # sorted, unique


def test_parse_channels_does_not_confuse_P1_and_P10(per_channel_X):
    channels = parse_channels_from_columns(per_channel_X.columns.tolist())
    assert "P1" in channels and "P10" in channels


def test_select_channel_columns_returns_only_that_channel(per_channel_X):
    X_p1 = select_channel_columns(per_channel_X, "P1")
    assert all(c.startswith("P1_") for c in X_p1.columns)
    # P10 columns must NOT leak into P1
    assert not any(c.startswith("P10_") for c in X_p1.columns)
    assert X_p1.shape[1] == 3  # 3 markers


def test_select_channel_columns_raises_on_unknown_channel(per_channel_X):
    with pytest.raises(ValueError, match="no columns"):
        select_channel_columns(per_channel_X, "Cz")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'parse_channels_from_columns'`.

- [ ] **Step 4: Implement parsing + selection**

`mw_classification_pipeline/utils/spatial_decoding_utils.py` (module header + first two functions):

```python
"""
Spatial decoding (per-electrode searchlight) core, shared by the LOSO and
Within-Subject MW classification pipelines.

Each per-electrode model is the existing classification engine restricted to a
single channel's marker columns. This module owns only the spatial orchestration:
channel parsing, the channel loop, multiple-comparisons correction, MNE montage
construction, and topomap rendering. The ML engine itself is supplied by the
caller as a `channel_eval` callable (see module docstring of each driver).

Column convention: per-channel feature columns are named ``{channel}_{marker}``.
"""

from __future__ import annotations

import os
import re
from typing import Callable

import numpy as np
import pandas as pd


def parse_channels_from_columns(columns: list[str]) -> list[str]:
    """
    Extract the sorted, unique set of channel names from ``{channel}_{marker}`` columns.

    The channel token is everything before the first underscore. This is robust to
    markers that themselves contain underscores (e.g. ``psd_bands_delta_mean``).

    Parameters
    ----------
    columns : list of str
        Feature column names in ``{channel}_{marker}`` form.

    Returns
    -------
    list of str
        Sorted unique channel names.
    """
    channels = {col.split("_", 1)[0] for col in columns if "_" in col}
    return sorted(channels)


def select_channel_columns(X: pd.DataFrame, channel: str) -> pd.DataFrame:
    """
    Restrict a per-channel feature matrix to a single channel's marker columns.

    Uses an exact ``{channel}_`` prefix so that ``P1`` does not match ``P10_*``.

    Parameters
    ----------
    X : pd.DataFrame
        Per-channel feature matrix.
    channel : str
        Electrode name.

    Returns
    -------
    pd.DataFrame
        Sub-matrix with only ``{channel}_*`` columns.

    Raises
    ------
    ValueError
        If the channel matches no columns.
    """
    prefix = f"{channel}_"
    cols = [c for c in X.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"Channel '{channel}' matched no columns in X.")
    return X[cols]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/__init__.py \
        mw_classification_pipeline/tests/conftest.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py
git commit -m "feat(spatial): add channel parsing and column selection for searchlight decoding"
```

---

## Task 2: FDR correction across channels

**Files:**
- Modify: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Test: `mw_classification_pipeline/tests/test_spatial_decoding.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_spatial_decoding.py`:

```python
from utils.spatial_decoding_utils import fdr_correct


def test_fdr_correct_matches_benjamini_hochberg_reference():
    # Known BH example: p = [0.01, 0.02, 0.03, 0.04, 0.05], n=5
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    p_adj, reject = fdr_correct(p, alpha=0.05)
    # BH adjusted: each p*(n/rank); monotone-enforced
    expected_adj = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(p_adj, expected_adj, rtol=1e-9)
    assert reject.tolist() == [True, True, True, True, True]


def test_fdr_correct_rejects_nothing_when_all_large():
    p = np.array([0.4, 0.6, 0.8])
    p_adj, reject = fdr_correct(p, alpha=0.05)
    assert reject.tolist() == [False, False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k fdr -v`
Expected: FAIL with `cannot import name 'fdr_correct'`.

- [ ] **Step 3: Implement FDR (Benjamini-Hochberg)**

Append to `utils/spatial_decoding_utils.py`:

```python
def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (1-D).
    alpha : float
        Target false-discovery rate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (adjusted_p_values, reject_mask) in the original input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    # BH-adjusted p in sorted order, with monotone (cumulative-min from the top) enforcement.
    adj_sorted = ranked * n / (np.arange(n) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    reject = adj <= alpha
    return adj, reject
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k fdr -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py
git commit -m "feat(spatial): add Benjamini-Hochberg FDR correction across channels"
```

---

## Task 3: Permutation p-value (+1 convention)

**Files:**
- Modify: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Test: `mw_classification_pipeline/tests/test_spatial_decoding.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_spatial_decoding.py`:

```python
from utils.spatial_decoding_utils import permutation_pvalue


def test_permutation_pvalue_plus_one_convention():
    # true=0.70, null has 9 values; 1 of them >= 0.70 -> p = (1+1)/(1+9) = 0.2
    null = [0.50, 0.55, 0.60, 0.45, 0.52, 0.58, 0.49, 0.71, 0.40]
    p = permutation_pvalue(true_value=0.70, null_values=null)
    assert p == pytest.approx((1 + 1) / (1 + 9))


def test_permutation_pvalue_floor_with_empty_null():
    # No null -> p = (1+0)/(1+0) = 1.0
    assert permutation_pvalue(0.7, []) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k permutation_pvalue -v`
Expected: FAIL with `cannot import name 'permutation_pvalue'`.

- [ ] **Step 3: Implement permutation p-value**

Append to `utils/spatial_decoding_utils.py`:

```python
def permutation_pvalue(true_value: float, null_values: list[float]) -> float:
    """
    One-sided permutation p-value with the +1 convention.

    ``p = (1 + #{null >= true}) / (1 + n_perm)``. This is the project-standard
    estimator (never returns exactly 0). Higher metric = better, so the test is
    right-tailed.

    Parameters
    ----------
    true_value : float
        Observed statistic (e.g. mean AUC).
    null_values : list of float
        Permutation null distribution of the same statistic.

    Returns
    -------
    float
        Permutation p-value in ``(0, 1]``.
    """
    null = np.asarray(null_values, dtype=float)
    null = null[~np.isnan(null)]
    n = null.size
    n_ge = int(np.sum(null >= true_value))
    return (1 + n_ge) / (1 + n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k permutation_pvalue -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py
git commit -m "feat(spatial): add +1-convention permutation p-value"
```

---

## Task 4: Searchlight channel-loop orchestrator

**Files:**
- Modify: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Test: `mw_classification_pipeline/tests/test_spatial_decoding.py`

- [ ] **Step 1: Write failing smoke test with a stub channel_eval**

Append to `tests/test_spatial_decoding.py`:

```python
from utils.spatial_decoding_utils import run_spatial_searchlight


def _stub_channel_eval(X_ch, channel, **kwargs):
    """Deterministic stub: AUC scales with number of features; null centered at 0.5."""
    rng = np.random.default_rng(abs(hash(channel)) % (2**32))
    mean_auc = 0.5 + 0.01 * X_ch.shape[1]
    null = (0.5 + 0.02 * rng.standard_normal(20)).tolist()
    return {
        "channel": channel,
        "n_features": X_ch.shape[1],
        "mean_auc": mean_auc,
        "std_auc": 0.01,
        "null_aucs": null,
        "subject_auc": None,
    }


def test_run_spatial_searchlight_returns_one_row_per_channel(per_channel_X, tmp_path):
    df = run_spatial_searchlight(
        X=per_channel_X,
        channel_eval=_stub_channel_eval,
        alpha=0.05,
        results_path=str(tmp_path),
    )
    assert sorted(df["channel"].tolist()) == ["Fz", "P1", "P10", "Pz"]
    for col in ["mean_auc", "std_auc", "perm_p", "perm_p_fdr", "sig", "n_features"]:
        assert col in df.columns
    # FDR p >= raw p elementwise
    assert (df["perm_p_fdr"] >= df["perm_p"] - 1e-9).all()
    # CSV persisted
    assert (tmp_path / "per_channel_metrics.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k searchlight -v`
Expected: FAIL with `cannot import name 'run_spatial_searchlight'`.

- [ ] **Step 3: Implement the orchestrator**

Append to `utils/spatial_decoding_utils.py`:

```python
def run_spatial_searchlight(
    X: pd.DataFrame,
    channel_eval: Callable[..., dict],
    alpha: float = 0.05,
    results_path: str | None = None,
    channels: list[str] | None = None,
    save_nulls: bool = True,
    verbose: bool = True,
    **eval_kwargs,
) -> pd.DataFrame:
    """
    Run a per-electrode searchlight: evaluate one model per channel and aggregate.

    For each channel, restricts ``X`` to that channel's marker columns and calls
    ``channel_eval`` (the pipeline-specific adapter). Computes the +1 permutation
    p-value per channel from the returned null distribution, applies BH-FDR across
    channels, and writes ``per_channel_metrics.csv`` (and optionally the per-channel
    null distributions) to ``results_path``.

    Parameters
    ----------
    X : pd.DataFrame
        Per-channel feature matrix (``{channel}_{marker}`` columns).
    channel_eval : callable
        Adapter returning the normalized result dict (see module/plan contract).
    alpha : float
        FDR significance threshold.
    results_path : str or None
        Output directory. If None, nothing is written to disk.
    channels : list of str or None
        Subset of channels to run. None = all channels parsed from columns.
    save_nulls : bool
        Persist per-channel null AUC distributions to ``permutation_nulls.csv``.
    verbose : bool
        Print per-channel progress.
    **eval_kwargs
        Forwarded to ``channel_eval``.

    Returns
    -------
    pd.DataFrame
        One row per channel with columns: channel, n_features, mean_auc, std_auc,
        perm_p, perm_p_fdr, sig, n_sig_subjects (if provided by the adapter).
    """
    if channels is None:
        channels = parse_channels_from_columns(X.columns.tolist())

    rows = []
    null_rows = []
    for i, ch in enumerate(channels):
        X_ch = select_channel_columns(X, ch)
        if verbose:
            print(f"  [{i + 1}/{len(channels)}] channel {ch}: {X_ch.shape[1]} features")
        res = channel_eval(X_ch, ch, **eval_kwargs)
        p = permutation_pvalue(res["mean_auc"], res.get("null_aucs", []))
        row = {
            "channel": ch,
            "n_features": res.get("n_features", X_ch.shape[1]),
            "mean_auc": res["mean_auc"],
            "std_auc": res.get("std_auc", np.nan),
            "perm_p": p,
        }
        subject_auc = res.get("subject_auc")
        if subject_auc is not None:
            row["n_sig_subjects"] = res.get("n_sig_subjects", np.nan)
        rows.append(row)
        if save_nulls:
            for v in res.get("null_aucs", []):
                null_rows.append({"channel": ch, "null_auc": v})

    metrics = pd.DataFrame(rows)
    adj, reject = fdr_correct(metrics["perm_p"].to_numpy(), alpha=alpha)
    metrics["perm_p_fdr"] = adj
    metrics["sig"] = reject

    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        metrics.to_csv(os.path.join(results_path, "per_channel_metrics.csv"), index=False)
        if save_nulls and null_rows:
            pd.DataFrame(null_rows).to_csv(
                os.path.join(results_path, "permutation_nulls.csv"), index=False
            )
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k searchlight -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py
git commit -m "feat(spatial): add searchlight channel-loop orchestrator with FDR"
```

---

## Task 5: MNE montage builder and topomap rendering

**Files:**
- Modify: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Test: `mw_classification_pipeline/tests/test_spatial_decoding.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_spatial_decoding.py`:

```python
from utils.spatial_decoding_utils import build_info_from_channels, plot_channel_topomap


def test_build_info_positions_known_1020_channels():
    info = build_info_from_channels(["Fz", "Cz", "Pz", "Oz"], montage="standard_1020")
    assert info["nchan"] == 4
    assert info["ch_names"] == ["Fz", "Cz", "Pz", "Oz"]


def test_plot_channel_topomap_writes_png(tmp_path):
    metrics = pd.DataFrame({
        "channel": ["Fz", "Cz", "Pz", "Oz"],
        "mean_auc": [0.55, 0.60, 0.70, 0.52],
        "perm_p_fdr": [0.20, 0.04, 0.01, 0.50],
        "sig": [False, True, True, False],
    })
    out = tmp_path / "topomap_auc.png"
    plot_channel_topomap(
        metrics, value_col="mean_auc", montage="standard_1020",
        out_path=str(out), mask_col=None, title="AUC",
    )
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k "info or topomap" -v`
Expected: FAIL with `cannot import name 'build_info_from_channels'`.

- [ ] **Step 3: Implement montage builder + topomap**

Append to `utils/spatial_decoding_utils.py`:

```python
def build_info_from_channels(channels: list[str], montage: str = "standard_1020"):
    """
    Build an MNE Info with positions from a standard montage for the given channels.

    Parameters
    ----------
    channels : list of str
        EEG channel names (must exist in the montage).
    montage : str
        MNE standard montage name.

    Returns
    -------
    mne.Info
        Info object with the montage applied, ready for ``plot_topomap``.
    """
    import mne

    info = mne.create_info(ch_names=list(channels), sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(montage), match_case=False)
    return info


def plot_channel_topomap(
    metrics: pd.DataFrame,
    value_col: str,
    out_path: str,
    montage: str = "standard_1020",
    mask_col: str | None = None,
    title: str = "",
    cmap: str = "RdBu_r",
    vlim: tuple | None = None,
) -> None:
    """
    Render a scalp topomap of a per-channel metric.

    Parameters
    ----------
    metrics : pd.DataFrame
        Per-channel metrics; must contain ``channel`` and ``value_col``.
    value_col : str
        Column to map (e.g. ``mean_auc``).
    out_path : str
        Output PNG path.
    montage : str
        MNE standard montage name.
    mask_col : str or None
        Boolean column used to mark significant electrodes.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap.
    vlim : tuple or None
        (vmin, vmax). None = auto.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    channels = metrics["channel"].tolist()
    info = build_info_from_channels(channels, montage=montage)
    values = metrics[value_col].to_numpy(dtype=float)
    mask = metrics[mask_col].to_numpy(dtype=bool) if mask_col else None

    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=cmap,
        vlim=(None, None) if vlim is None else vlim,
        mask=mask,
        mask_params=dict(marker="o", markerfacecolor="k", markersize=6),
    )
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass** (in the `eeg` env, which has MNE)

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k "info or topomap" -v`
Expected: 2 passed. If MNE is missing in the active env, run with the `eeg` env Python.

- [ ] **Step 5: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py
git commit -m "feat(spatial): add MNE montage builder and topomap rendering"
```

---

## Task 6: LOSO driver, adapter, config, merge, SLURM

**Files:**
- Create: `mw_classification_pipeline/loso_pipeline/spatial_decoding/run_loso_spatial_decoding.py`
- Create: `mw_classification_pipeline/loso_pipeline/spatial_decoding/config.yaml`
- Create: `mw_classification_pipeline/loso_pipeline/spatial_decoding/merge_spatial_results.py`
- Create: `mw_classification_pipeline/loso_pipeline/spatial_decoding/run_spatial_slurm.sh`

- [ ] **Step 1: Create the LOSO spatial config**

`loso_pipeline/spatial_decoding/config.yaml` — copy `loso_pipeline/config.yaml` verbatim, then change/add only:

```yaml
# --- Spatial decoding overrides (everything else inherited from the LOSO config) ---
data_format: "per_channel"     # REQUIRED: searchlight needs channel-level columns

spatial_decoding:
  channels: "all"              # or explicit list to subset electrodes
  montage: "standard_1020"
  feature_selection_k: 10      # mRMR k per-channel model (smaller pool than main pipeline)
  multiple_comparisons:
    method: "fdr_bh"
    alpha: 0.05
  topomap:
    cmap: "RdBu_r"
    vlim: null                 # or [0.4, 0.7] to fix the AUC color scale
  contrasts:                   # the 5 dimensions to map
    - ON_vs_OFF_within_median
    - valence_within_median
    - selfother_within_median
    - time_within_median
    - confidence_within_median
```

- [ ] **Step 2: Implement the LOSO driver + adapter**

`loso_pipeline/spatial_decoding/run_loso_spatial_decoding.py`:

```python
#!/usr/bin/env python
"""
LOSO spatial decoding driver.

Runs a per-electrode searchlight for one (contrast, channel) cell using the
existing LOSO engine restricted to that channel's columns. Designed for SLURM
array execution: one task per (contrast, channel). Aggregate with
merge_spatial_results.py to build topomaps.

USAGE:
    python run_loso_spatial_decoding.py --config config.yaml \
        --contrast ON_vs_OFF_within_median --channel Pz
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # mw_classification_pipeline/

from utils.data_utils import load_config, prepare_data_for_contrast, get_project_root
from utils.analysis_utils import (
    run_distribution_analysis,
    run_permutation_distribution_analysis,
)
from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    select_channel_columns,
    run_spatial_searchlight,
)


def build_results_path(config: dict, contrast: str) -> str:
    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(root / results_root)
    family = config.get("active_family", "all")
    model = config.get("model_type", "rf")
    path = os.path.join(results_root, "SpatialDecoding", "LOSO", contrast, family, model)
    os.makedirs(path, exist_ok=True)
    return path


def make_loso_channel_eval(df, y, groups, contrast, config):
    """Return a channel_eval adapter bound to the loaded LOSO data."""
    sd = config["spatial_decoding"]
    k = sd.get("feature_selection_k", 10)
    n_runs = config.get("n_runs", 20)
    n_perm = config.get("permutation_runs", 100)

    def channel_eval(X_ch, channel, **_):
        feature_cols = X_ch.columns.tolist()
        results_df, _all, _shap = run_distribution_analysis(
            dimension=contrast, df=df, X=X_ch, y=y, groups=groups,
            feature_cols=feature_cols, config=config,
            n_runs=n_runs, results_path=os.devnull, model_type=config.get("model_type", "rf"),
            use_smote=config.get("use_smote", False),
            oversampling_method=config.get("oversampling_method", "SMOTE"),
            oversampling_scope=config.get("oversampling_scope", "within"),
            k=k, feature_selection_method=config.get("feature_selection_method", "mrmr"),
            scaler=config.get("scaler", "standard"),
            save_pickle=False, save_csv=False, save_probabilities=False,
            save_plots=False, save_shap=False, verbose=False,
            cv_n_jobs=config.get("parallelism", {}).get("true_cv_n_jobs", 1),
        )
        true_aucs = results_df["mean_auc"].dropna().tolist()
        mean_auc = float(sum(true_aucs) / len(true_aucs)) if true_aucs else float("nan")

        perm_df, _summ, _pall, _pshap = run_permutation_distribution_analysis(
            dimension=contrast, df=df, X=X_ch, y=y, groups=groups,
            feature_cols=feature_cols, config=config,
            n_permutations=n_perm, results_path=os.devnull,
            model_type=config.get("model_type", "rf"),
            use_smote=config.get("use_smote", False),
            oversampling_method=config.get("oversampling_method", "SMOTE"),
            oversampling_scope=config.get("oversampling_scope", "within"),
            k=k, feature_selection_method=config.get("feature_selection_method", "mrmr"),
            scaler=config.get("scaler", "standard"),
            true_auc_list=true_aucs,
            save_pickle=False, save_csv=False, save_probabilities=False,
            save_plots=False, save_shap=False, verbose=False,
            permutation_scope=config.get("permutation_scope", "within"),
        )
        null_aucs = perm_df["mean_auc"].dropna().tolist() if "mean_auc" in perm_df else []
        import numpy as np
        return {
            "channel": channel, "n_features": X_ch.shape[1],
            "mean_auc": mean_auc, "std_auc": float(np.std(true_aucs)) if true_aucs else float("nan"),
            "null_aucs": null_aucs, "subject_auc": None,
        }
    return channel_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--channel", default=None, help="Single channel (SLURM array). None = all.")
    args = ap.parse_args()

    config = load_config(args.config)
    config["contrast"] = args.contrast
    sd = config["spatial_decoding"]
    config["epoch_types"] = config["feature_families"][config.get("active_family", "all")]["epoch_types"]

    df, X, y, groups, _cols = prepare_data_for_contrast(config, args.contrast, verbose=False)

    all_channels = parse_channels_from_columns(X.columns.tolist())
    if sd.get("channels", "all") != "all":
        all_channels = [c for c in all_channels if c in sd["channels"]]
    channels = [args.channel] if args.channel else all_channels

    results_path = build_results_path(config, args.contrast)
    with open(os.path.join(results_path, "used_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    channel_eval = make_loso_channel_eval(df, y, groups, args.contrast, config)
    # In single-channel SLURM mode, write to a per-channel shard for later merge.
    shard = results_path if args.channel is None else os.path.join(results_path, "shards")
    metrics = run_spatial_searchlight(
        X=X, channel_eval=channel_eval,
        alpha=sd["multiple_comparisons"]["alpha"],
        results_path=None, channels=channels, verbose=True,
    )
    if args.channel is not None:
        os.makedirs(shard, exist_ok=True)
        metrics.to_csv(os.path.join(shard, f"channel-{args.channel}.csv"), index=False)
        print(f"Shard written: {shard}/channel-{args.channel}.csv")
    else:
        metrics.to_csv(os.path.join(results_path, "per_channel_metrics.csv"), index=False)
        print(f"Full metrics written: {results_path}/per_channel_metrics.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the driver runs on a tiny channel subset (dry smoke)**

Run (ML env):
```bash
cd mw_classification_pipeline
~/miniforge3/envs/ML/bin/python loso_pipeline/spatial_decoding/run_loso_spatial_decoding.py \
  --config loso_pipeline/spatial_decoding/config.yaml \
  --contrast ON_vs_OFF_within_median --channel Pz
```
Expected: prints feature count for `Pz`, writes `.../SpatialDecoding/LOSO/ON_vs_OFF_within_median/all/rf/shards/channel-Pz.csv` with one row containing `mean_auc` and `perm_p`. **Confirm `perm_df` exposes a `mean_auc` column**; if the permutation function names the null AUC column differently, adjust the `null_aucs` extraction line and re-run.

- [ ] **Step 4: Implement the merge script**

`loso_pipeline/spatial_decoding/merge_spatial_results.py`:

```python
#!/usr/bin/env python
"""Merge per-channel SLURM shards into per_channel_metrics.csv + topomaps (LOSO)."""
import os
import sys
import argparse
import glob
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.spatial_decoding_utils import fdr_correct, plot_channel_topomap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="…/SpatialDecoding/LOSO/{contrast}/{family}/{model}")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--montage", default="standard_1020")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.results_dir, "shards", "channel-*.csv")))
    if not shards:
        raise FileNotFoundError(f"No shards in {args.results_dir}/shards")
    metrics = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)

    # Re-derive FDR across the full channel set (shards each carried a single-channel p).
    adj, reject = fdr_correct(metrics["perm_p"].to_numpy(), alpha=args.alpha)
    metrics["perm_p_fdr"] = adj
    metrics["sig"] = reject
    metrics = metrics.sort_values("channel").reset_index(drop=True)
    metrics.to_csv(os.path.join(args.results_dir, "per_channel_metrics.csv"), index=False)

    plot_channel_topomap(metrics, "mean_auc",
                         os.path.join(args.results_dir, "topomap_auc.png"),
                         montage=args.montage, title="Spatial decoding AUC")
    plot_channel_topomap(metrics, "mean_auc",
                         os.path.join(args.results_dir, "topomap_sig.png"),
                         montage=args.montage, mask_col="sig",
                         title="AUC (FDR-significant electrodes marked)")
    print(f"Merged {len(metrics)} channels → {args.results_dir}/per_channel_metrics.csv + topomaps")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement the SLURM array script**

`loso_pipeline/spatial_decoding/run_spatial_slurm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=loso_spatial
#SBATCH --output=logs/loso_spatial_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# 5 contrasts × 64 channels = 320 tasks. array_idx = contrast_idx*64 + channel_idx
#SBATCH --array=0-319

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(ON_vs_OFF_within_median valence_within_median selfother_within_median time_within_median confidence_within_median)
mapfile -t CHANNELS < <("$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
sys.path.insert(0, __import__("pathlib").Path("'"$HERE"'").parents[1].as_posix())
# Channels come from the data; list the canonical 64-channel 10-20 set used in this dataset.
CH = ['AF3','AF4','AF7','AF8','AFz','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','FT10','FT7','FT8','FT9','Fp1','Fp2','Fz','Iz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP10','TP7','TP8','TP9']
print("\n".join(CH))
PY
)

N_CH=${#CHANNELS[@]}
CONTRAST_IDX=$(( SLURM_ARRAY_TASK_ID / N_CH ))
CHANNEL_IDX=$(( SLURM_ARRAY_TASK_ID % N_CH ))
CONTRAST=${CONTRASTS[$CONTRAST_IDX]}
CHANNEL=${CHANNELS[$CHANNEL_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST channel=$CHANNEL"
"$PYTHON" "$HERE/run_loso_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST" --channel "$CHANNEL"
```

- [ ] **Step 6: Commit**

```bash
git add mw_classification_pipeline/loso_pipeline/spatial_decoding/
git commit -m "feat(spatial): add LOSO spatial decoding driver, config, merge, and SLURM array"
```

---

## Task 7: Within-Subject driver, adapter, config, merge, SLURM

**Files:**
- Create: `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/run_within_spatial_decoding.py`
- Create: `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/config.yaml`
- Create: `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/merge_spatial_results.py`
- Create: `mw_classification_pipeline/within_subject_pipeline/spatial_decoding/run_spatial_slurm.sh`

- [ ] **Step 1: Create the WS spatial config**

`within_subject_pipeline/spatial_decoding/config.yaml` — copy `within_subject_pipeline/config.yaml` verbatim, then change/add only the same `data_format: per_channel` line and `spatial_decoding:` block as in Task 6 Step 1 (identical block; `contrasts` uses the WS contrast names: `on_vs_off_within_median`, `valence_within_median`, `selfother_within_median`, `time_within_median`, `confidence_within_median`).

- [ ] **Step 2: Implement the WS driver + adapter**

`within_subject_pipeline/spatial_decoding/run_within_spatial_decoding.py`:

```python
#!/usr/bin/env python
"""
Within-Subject spatial decoding driver.

Per-electrode searchlight using the existing within-subject engine restricted to
one channel's columns. The per-channel statistic is the GROUP-MEAN AUC across
subjects; the permutation null is the group-mean AUC per permutation. Also records
the count of individually-significant subjects per channel.

USAGE:
    python run_within_spatial_decoding.py --config config.yaml \
        --contrast on_vs_off_within_median --channel Pz
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import load_config, prepare_data_for_contrast, get_project_root
from utils.analysis_utils import (
    run_within_subject_distribution_analysis,
    run_within_subject_permutation_analysis,
)
from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    run_spatial_searchlight,
)


def build_results_path(config: dict, contrast: str) -> str:
    results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(root / results_root)
    family = config.get("active_family", "all")
    model = config.get("model_type", "rf")
    path = os.path.join(results_root, "SpatialDecoding", "WithinSubject", contrast, family, model)
    os.makedirs(path, exist_ok=True)
    return path


def make_ws_channel_eval(df, y, groups, tasks, contrast, config):
    """Return a channel_eval adapter bound to the loaded WS data."""
    sd = config["spatial_decoding"]
    k = sd.get("feature_selection_k", 10)
    n_runs = config.get("n_runs", 100)
    n_perm = config.get("n_permutations", 500)
    cv = config.get("cv", {})
    osc = config.get("oversampling", {})
    cls = config.get("classifiers", {})
    model = config.get("model_type", "rf")

    common = dict(
        df=df, y=y, subjects=groups, tasks=tasks, config=config, model_type=model,
        use_smote=osc.get("use_smote", False), oversampling_method=osc.get("method", "SMOTE"),
        oversampling_scope=osc.get("scope", "within"),
        cv_strategy=cv.get("strategy", "stratified_kfold"), cv_folds=cv.get("folds", 5),
        min_samples_per_class=cv.get("min_samples_per_class", 0),
        min_minority_ratio=config.get("min_minority_ratio", 0.0),
        k=k, rf_params=cls.get("rf", {}),
        feature_selection_method=config.get("feature_selection", {}).get("method", "mrmr"),
        scaler=config.get("preprocessing", {}).get("scaler", "standard"),
        save_pickle=False, save_csv=False, save_probabilities=False,
        save_plots=False, save_shap=False, verbose=False,
    )

    def channel_eval(X_ch, channel, **_):
        from utils.spatial_decoding_utils import select_channel_columns  # noqa
        feature_cols = X_ch.columns.tolist()
        true_metrics, _shap = run_within_subject_distribution_analysis(
            dimension=contrast, X=X_ch, feature_cols=feature_cols,
            n_runs=n_runs, results_path=os.devnull, **common,
        )
        subj_df = pd.DataFrame(true_metrics)
        subject_auc = subj_df.groupby("subject")["mean_auc"].mean().to_dict()
        group_mean_auc = float(np.mean(list(subject_auc.values()))) if subject_auc else float("nan")

        perm_df, perm_summary, _pall, _pshap = run_within_subject_permutation_analysis(
            dimension=contrast, X=X_ch, feature_cols=feature_cols,
            n_permutations=n_perm, results_path=os.devnull,
            true_ws_metrics_df=subj_df, **common,
        )
        # Group-mean AUC per permutation = the null for the group statistic.
        if {"perm_idx", "mean_auc"}.issubset(perm_df.columns):
            null_aucs = perm_df.groupby("perm_idx")["mean_auc"].mean().tolist()
        else:
            null_aucs = perm_df["mean_auc"].dropna().tolist() if "mean_auc" in perm_df else []

        # Count individually-significant subjects from the permutation summary if present.
        n_sig = int(perm_summary.get("n_sig_subjects", np.nan)) if isinstance(perm_summary, dict) else np.nan

        return {
            "channel": channel, "n_features": X_ch.shape[1],
            "mean_auc": group_mean_auc,
            "std_auc": float(subj_df.groupby("subject")["mean_auc"].mean().std()) if subject_auc else float("nan"),
            "null_aucs": null_aucs, "subject_auc": subject_auc, "n_sig_subjects": n_sig,
        }
    return channel_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--channel", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    config["contrast"] = args.contrast
    sd = config["spatial_decoding"]
    config["epoch_types"] = config["feature_families"][config.get("active_family", "all")]["epoch_types"]

    df, X, y, groups, _cols = prepare_data_for_contrast(config, args.contrast, verbose=False)
    tasks = df["task"] if "task" in df.columns else groups

    all_channels = parse_channels_from_columns(X.columns.tolist())
    if sd.get("channels", "all") != "all":
        all_channels = [c for c in all_channels if c in sd["channels"]]
    channels = [args.channel] if args.channel else all_channels

    results_path = build_results_path(config, args.contrast)
    with open(os.path.join(results_path, "used_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    channel_eval = make_ws_channel_eval(df, y, groups, tasks, args.contrast, config)
    metrics = run_spatial_searchlight(
        X=X, channel_eval=channel_eval,
        alpha=sd["multiple_comparisons"]["alpha"],
        results_path=None, channels=channels, verbose=True,
    )
    shard = os.path.join(results_path, "shards")
    if args.channel is not None:
        os.makedirs(shard, exist_ok=True)
        metrics.to_csv(os.path.join(shard, f"channel-{args.channel}.csv"), index=False)
        print(f"Shard written: {shard}/channel-{args.channel}.csv")
    else:
        metrics.to_csv(os.path.join(results_path, "per_channel_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the WS driver on one channel (dry smoke)**

Run (ML env):
```bash
cd mw_classification_pipeline
~/miniforge3/envs/ML/bin/python within_subject_pipeline/spatial_decoding/run_within_spatial_decoding.py \
  --config within_subject_pipeline/spatial_decoding/config.yaml \
  --contrast on_vs_off_within_median --channel Pz
```
Expected: writes `.../SpatialDecoding/WithinSubject/on_vs_off_within_median/all/rf/shards/channel-Pz.csv` with `mean_auc`, `perm_p`, `n_sig_subjects`. **Confirm `perm_df` column names** for the null extraction; adjust the `null_aucs` branch if the permutation function uses different column names, then re-run.

- [ ] **Step 4: Implement the WS merge script**

`within_subject_pipeline/spatial_decoding/merge_spatial_results.py`: identical to the LOSO merge (Task 6 Step 4) except the docstring says "(WS)". Copy that file and change only the docstring line. The script is dimension-agnostic — it reads shards, re-derives FDR, and renders `topomap_auc.png` + `topomap_sig.png`.

- [ ] **Step 5: Implement the WS SLURM array**

`within_subject_pipeline/spatial_decoding/run_spatial_slurm.sh`: copy the LOSO SLURM script (Task 6 Step 5) and change only:
- `#SBATCH --job-name=ws_spatial` and the `--output` log prefix,
- the `CONTRASTS=(...)` array to the WS names: `on_vs_off_within_median valence_within_median selfother_within_median time_within_median confidence_within_median`,
- the invoked script to `run_within_spatial_decoding.py`.

- [ ] **Step 6: Commit**

```bash
git add mw_classification_pipeline/within_subject_pipeline/spatial_decoding/
git commit -m "feat(spatial): add within-subject spatial decoding driver, config, merge, and SLURM array"
```

---

## Task 8: Combined 5-dimension topomap panel

**Files:**
- Modify: `mw_classification_pipeline/utils/spatial_decoding_utils.py`
- Test: `mw_classification_pipeline/tests/test_spatial_decoding.py`
- Create: `mw_classification_pipeline/scripts/generate_spatial_panel.py`

- [ ] **Step 1: Write failing test for the panel function**

Append to `tests/test_spatial_decoding.py`:

```python
from utils.spatial_decoding_utils import plot_combined_topomap_panel


def test_plot_combined_panel_writes_png(tmp_path):
    chans = ["Fz", "Cz", "Pz", "Oz"]
    per_dim = {
        d: pd.DataFrame({"channel": chans,
                         "mean_auc": [0.55, 0.6, 0.7, 0.52],
                         "sig": [False, True, True, False]})
        for d in ["onoff", "valence", "selfother", "time", "confidence"]
    }
    out = tmp_path / "panel.png"
    plot_combined_topomap_panel(per_dim, value_col="mean_auc", mask_col="sig",
                                out_path=str(out), montage="standard_1020")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k panel -v`
Expected: FAIL with `cannot import name 'plot_combined_topomap_panel'`.

- [ ] **Step 3: Implement the panel function**

Append to `utils/spatial_decoding_utils.py`:

```python
def plot_combined_topomap_panel(
    per_dimension_metrics: dict,
    value_col: str,
    out_path: str,
    montage: str = "standard_1020",
    mask_col: str | None = None,
    cmap: str = "RdBu_r",
    vlim: tuple | None = None,
) -> None:
    """
    Render a single figure with one topomap per dimension (paper figure).

    Parameters
    ----------
    per_dimension_metrics : dict[str, pd.DataFrame]
        Maps dimension name -> per-channel metrics DataFrame (channel + value_col).
    value_col : str
        Metric column to map.
    out_path : str
        Output PNG path.
    montage : str
        MNE standard montage name.
    mask_col : str or None
        Boolean column marking significant electrodes.
    cmap : str
        Colormap.
    vlim : tuple or None
        Shared (vmin, vmax) across panels. None = auto per panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    dims = list(per_dimension_metrics.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(3 * len(dims), 3.2))
    if len(dims) == 1:
        axes = [axes]
    im = None
    for ax, dim in zip(axes, dims):
        m = per_dimension_metrics[dim]
        info = build_info_from_channels(m["channel"].tolist(), montage=montage)
        mask = m[mask_col].to_numpy(dtype=bool) if mask_col else None
        im, _ = mne.viz.plot_topomap(
            m[value_col].to_numpy(dtype=float), info, axes=ax, show=False, cmap=cmap,
            vlim=(None, None) if vlim is None else vlim, mask=mask,
            mask_params=dict(marker="o", markerfacecolor="k", markersize=5),
        )
        ax.set_title(dim)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, label=value_col)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -k panel -v`
Expected: 1 passed.

- [ ] **Step 5: Implement the panel-generation script**

`scripts/generate_spatial_panel.py`:

```python
#!/usr/bin/env python
"""Build the combined 5-dimension topomap panel from per-dimension metrics."""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.spatial_decoding_utils import plot_combined_topomap_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_dir", required=True,
                    help="…/SpatialDecoding/{LOSO|WithinSubject}")
    ap.add_argument("--family", default="all")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--value_col", default="mean_auc")
    ap.add_argument("--montage", default="standard_1020")
    args = ap.parse_args()

    per_dim = {}
    for contrast_dir in sorted(Path(args.pipeline_dir).iterdir()):
        if not contrast_dir.is_dir() or contrast_dir.name == "combined":
            continue
        metrics_csv = contrast_dir / args.family / args.model / "per_channel_metrics.csv"
        if metrics_csv.exists():
            per_dim[contrast_dir.name] = pd.read_csv(metrics_csv)

    if not per_dim:
        raise FileNotFoundError(f"No per_channel_metrics.csv under {args.pipeline_dir}")

    out = os.path.join(args.pipeline_dir, "combined", "topomap_panel_auc.png")
    plot_combined_topomap_panel(per_dim, value_col=args.value_col, mask_col="sig",
                                out_path=out, montage=args.montage)
    print(f"Panel written: {out}  ({len(per_dim)} dimensions)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test module**

Run: `cd mw_classification_pipeline && python -m pytest tests/test_spatial_decoding.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add mw_classification_pipeline/utils/spatial_decoding_utils.py \
        mw_classification_pipeline/tests/test_spatial_decoding.py \
        mw_classification_pipeline/scripts/generate_spatial_panel.py
git commit -m "feat(spatial): add combined 5-dimension topomap panel"
```

---

## Post-implementation validation (manual, after a real SLURM run)

1. Submit LOSO array: `sbatch loso_pipeline/spatial_decoding/run_spatial_slurm.sh`; after completion run `merge_spatial_results.py --results_dir …/<contrast>/all/rf` per contrast.
2. Submit WS array likewise.
3. Build panels: `scripts/generate_spatial_panel.py --pipeline_dir …/SpatialDecoding/LOSO` and `…/WithinSubject`.
4. **Scientific self-check:** AUC topomaps for `onoff` should show maxima over central/parietal sites (P3-like attentional signature, consistent with CBPT Section 2). Verify the spatial decodability ordering across dimensions mirrors the CBPT signal-density hierarchy (onoff > valence > …). Flag any inversion as a finding to investigate, not to silently accept.

---

## Self-Review notes

- **Spec coverage:** searchlight per electrode (Tasks 1,4,6,7); 5 dimensions (configs + SLURM in 6,7); per-electrode permutation (adapters 6,7 + p-value Task 3); FDR per dimension (Tasks 2,4, merge in 6,7); topomaps (Task 5) + combined panel (Task 8); per_channel outputs + suppressed artifacts (`save_*=False` in adapters); SLURM dim×channel axis (6,7); mRMR k=10 configurable (`feature_selection_k` in configs). All covered.
- **Open risk (flagged in Tasks 6.3 and 7.3):** the exact column name of the permutation null AUC in `perm_df` must be confirmed against `run_permutation_distribution_analysis` / `run_within_subject_permutation_analysis` at implementation time. The smoke-test steps verify this before the full SLURM submission and instruct the adjustment if names differ. This is the only place the plan depends on an internal column name.
