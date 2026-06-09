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


from utils.spatial_decoding_utils import permutation_pvalue


def test_permutation_pvalue_plus_one_convention():
    # true=0.70, null has 9 values; 1 of them >= 0.70 -> p = (1+1)/(1+9) = 0.2
    null = [0.50, 0.55, 0.60, 0.45, 0.52, 0.58, 0.49, 0.71, 0.40]
    p = permutation_pvalue(true_value=0.70, null_values=null)
    assert p == pytest.approx((1 + 1) / (1 + 9))


def test_permutation_pvalue_floor_with_empty_null():
    # No null -> p = (1+0)/(1+0) = 1.0
    assert permutation_pvalue(0.7, []) == pytest.approx(1.0)


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


from utils.spatial_decoding_utils import spatial_cache_path


def test_spatial_cache_path_includes_data_format_and_is_collision_safe(tmp_path):
    p_roi = spatial_cache_path(str(tmp_path), "ON_vs_OFF_within_median", "all", "per_roi")
    p_ch = spatial_cache_path(str(tmp_path), "ON_vs_OFF_within_median", "all", "per_channel")
    # per_roi and per_channel caches must NOT collide
    assert p_roi != p_ch
    assert p_ch.endswith("ON_vs_OFF_within_median__all__per_channel.pkl")
    # slashes in contrast names are sanitised
    p_slash = spatial_cache_path(str(tmp_path), "a/b", "all", "per_channel")
    assert "/a_b__" in p_slash or p_slash.endswith("a_b__all__per_channel.pkl")
