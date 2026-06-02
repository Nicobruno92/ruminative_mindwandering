"""
Unit tests for the ``restrict_to`` pre-binarization filter in
``utils.data_utils.create_label_contrast``.

The filter lets content dimensions (valence, selfother, time, confidence) be
classified only within a restricted subset of probes (e.g. off-task /
mind-wandering), reusing the same binarization logic to define the subset.
All data is synthetic — no external files required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data_utils import create_label_contrast


def _make_df() -> pd.DataFrame:
    """Two subjects, each with a spread of onoff and valence ratings."""
    rows = []
    # Subject 02: onoff median = 50; valence median (off-task only) is computed
    # on the kept subset.
    for onoff, valence in [(10, 20), (30, 40), (50, 60), (70, 80), (90, 95)]:
        rows.append({"subject": "02", "onoff": onoff, "valence": valence})
    for onoff, valence in [(5, 10), (25, 30), (45, 55), (75, 85), (95, 90)]:
        rows.append({"subject": "03", "onoff": onoff, "valence": valence})
    return pd.DataFrame(rows)


def test_restrict_to_keeps_only_offtask_probes():
    """restrict_to keep='negative' must drop on-task probes before binarizing."""
    df = _make_df()
    contrast = {
        "column_name": "valence",
        "split_method": "within_subject_median",
        "gap": 0,
        "positive_above": True,
        "restrict_to": {
            "column_name": "onoff",
            "split_method": "within_subject_median",
            "keep": "negative",  # off-task = onoff below subject median
        },
    }
    out = create_label_contrast(df, contrast)

    # Off-task = onoff strictly below subject median (gap=0, positive_above=True
    # keeps >= median as positive, so below-median rows are the negative side).
    # Subject 02 median(onoff)=50 -> off-task onoff in {10,30}; subject 03
    # median(onoff)=45 -> off-task onoff in {5,25}. => 4 probes total.
    assert len(out) == 4
    assert set(out["onoff"]) == {10, 30, 5, 25}
    assert "target" in out.columns


def test_restrict_to_absent_keeps_all_probes():
    """Without restrict_to, all probes are binarized (current behavior)."""
    df = _make_df()
    contrast = {
        "column_name": "valence",
        "split_method": "within_subject_median",
        "gap": 0,
        "positive_above": True,
    }
    out = create_label_contrast(df, contrast)
    assert len(out) == len(df)


def test_restrict_to_median_computed_on_subset():
    """Valence median must be computed on the off-task subset, not all probes."""
    df = _make_df()
    contrast = {
        "column_name": "valence",
        "split_method": "within_subject_median",
        "gap": 0,
        "positive_above": True,
        "restrict_to": {
            "column_name": "onoff",
            "split_method": "within_subject_median",
            "keep": "negative",
        },
    }
    out = create_label_contrast(df, contrast)

    # Subject 02 off-task valence = {20, 40}, median = 30 -> 20<30 ->0, 40>=30 ->1
    sub02 = out[out["subject"] == "02"].sort_values("valence")
    assert list(sub02["target"]) == [0, 1]
    # Subject 03 off-task valence = {10, 30}, median = 20 -> 10<20 ->0, 30>=20 ->1
    sub03 = out[out["subject"] == "03"].sort_values("valence")
    assert list(sub03["target"]) == [0, 1]


def test_restrict_to_invalid_keep_raises():
    """An invalid 'keep' value must fail loudly (no silent fallback)."""
    df = _make_df()
    contrast = {
        "column_name": "valence",
        "split_method": "within_subject_median",
        "restrict_to": {
            "column_name": "onoff",
            "split_method": "within_subject_median",
            "keep": "middle",
        },
    }
    try:
        create_label_contrast(df, contrast)
    except ValueError as exc:
        assert "keep" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid restrict_to.keep")
