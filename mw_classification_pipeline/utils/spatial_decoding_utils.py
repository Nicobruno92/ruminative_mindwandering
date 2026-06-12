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
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Standard 10-20 montage for the topomaps. The project's CACS-64_REF.bvef has no
# fiducials, so MNE falls back to an identity head transform and the scalp comes out
# oval/misplaced; the standard_1020 template (built by name, with fiducials) gives the
# clean centred circular head used in the figures.
DEFAULT_MONTAGE = "standard_1020"


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


def permute_labels_within_subject(y: pd.Series, groups: pd.Series, seed: int) -> pd.Series:
    """
    Permute binary labels independently within each subject.

    Within-subject permutation preserves each subject's class composition (and the
    LOSO/within-subject structure) while breaking the label–feature association — the
    correct null for the max-statistic spatial test. The same shuffle (fixed ``seed``)
    must be applied to every channel within one permutation draw.

    Parameters
    ----------
    y : pd.Series
        Binary labels, positionally aligned to the feature matrix.
    groups : pd.Series
        Subject ID per sample, aligned to ``y``.
    seed : int
        Permutation seed (typically the permutation index).

    Returns
    -------
    pd.Series
        Permuted labels with the original index/order preserved.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups)
    out = y_arr.copy()
    # Iterate subjects in a fixed order so the permutation is fully deterministic.
    for subject in sorted(pd.unique(g_arr).tolist()):
        idx = np.where(g_arr == subject)[0]
        out[idx] = y_arr[idx][rng.permutation(len(idx))]
    return pd.Series(out, index=y.index)


def maxstat_pvalues(true_auc_by_channel: dict, max_null: np.ndarray,
                    alpha: float = 0.05) -> pd.DataFrame:
    """
    Family-wise (FWER) per-channel p-values from a max-statistic permutation null.

    Each permutation contributes one ``max over channels`` AUC; ``max_null`` is that
    distribution. An electrode is significant iff its true AUC exceeds the
    ``(1-alpha)`` quantile of the max-null. The per-channel FWER p-value uses the +1
    convention: ``p_c = (1 + #{max_null >= true_c}) / (1 + N)``.

    Parameters
    ----------
    true_auc_by_channel : dict[str, float]
        Per-channel true (n_runs-averaged) AUC.
    max_null : np.ndarray
        Max-over-channels AUC for each permutation draw.
    alpha : float
        FWER level.

    Returns
    -------
    pd.DataFrame
        Columns: channel, mean_auc, perm_p, sig, fwer_threshold.
    """
    null = np.asarray(max_null, dtype=float)
    null = null[~np.isnan(null)]
    n = null.size
    threshold = float(np.quantile(null, 1.0 - alpha)) if n else float("nan")

    rows = []
    for channel, true_auc in true_auc_by_channel.items():
        n_ge = int(np.sum(null >= true_auc))
        p = (1 + n_ge) / (1 + n)
        rows.append({
            "channel": channel,
            "mean_auc": true_auc,
            "perm_p": p,
            "sig": p < alpha,
            "fwer_threshold": threshold,
        })
    return pd.DataFrame(rows)


def spatial_cache_path(cache_dir: str, contrast: str, family: str, data_format: str) -> str:
    """
    Build a cache file path keyed by (contrast, family, data_format).

    Including ``data_format`` keeps per-channel caches from colliding with the
    per-roi caches written by the main pipelines (same contrast/family names).

    Parameters
    ----------
    cache_dir : str
        Directory holding cache pickles.
    contrast : str
        Contrast name (slashes are sanitised).
    family : str
        Feature family name.
    data_format : str
        ``per_channel`` or ``per_roi``.

    Returns
    -------
    str
        Absolute-or-relative cache file path.
    """
    safe = contrast.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}__{family}__{data_format}.pkl")


def load_or_prepare_data(config: dict, contrast: str, family: str, prefixes, cache_dir: str,
                         verbose: bool = False):
    """
    Load a cached per-channel dataset or build it once and cache it.

    On a cache miss this calls ``prepare_data_for_contrast`` (slow: reads many CSVs
    and pivots to wide form), filters columns to the family prefixes, and writes a
    pickle so subsequent runs (and SLURM jobs) load in under a second.

    Parameters
    ----------
    config : dict
        Pipeline configuration (must define ``data_format`` and ``epoch_types``).
    contrast : str
        Contrast name.
    family : str
        Feature family name.
    prefixes : list[str] or None
        Family column prefixes (None keeps all columns).
    cache_dir : str
        Directory for cache pickles.
    verbose : bool
        Verbose data loading.

    Returns
    -------
    tuple
        (df, X, y, groups, feature_cols)
    """
    import pickle
    import re
    from utils.data_utils import prepare_data_for_contrast

    data_format = config.get("data_format", "per_channel")
    os.makedirs(cache_dir, exist_ok=True)
    path = spatial_cache_path(cache_dir, contrast, family, data_format)

    if os.path.exists(path):
        with open(path, "rb") as f:
            c = pickle.load(f)
        return c["df"], c["X"], c["y"], c["groups"], c["feature_cols"]

    df, X, y, groups, feature_cols = prepare_data_for_contrast(config, contrast, verbose=verbose)

    if prefixes is not None:
        def _matches(col, prefix):
            p = re.escape(prefix)
            return bool(re.search(rf"_{p}(_|$)", col)) or bool(re.match(rf"{p}(_|$)", col))
        selected = [c for c in X.columns if any(_matches(c, p) for p in prefixes)]
        X = X[selected]
    feature_cols = X.columns.tolist()

    with open(path, "wb") as f:
        pickle.dump({"df": df, "X": X, "y": y, "groups": groups,
                     "feature_cols": feature_cols}, f, protocol=5)
    return df, X, y, groups, feature_cols


def load_true_per_channel(results_dir: str) -> "pd.DataFrame":
    """Load and de-duplicate per-channel true AUC from ``true/*.csv`` (shards or combined)."""
    import glob
    files = sorted(glob.glob(os.path.join(results_dir, "true", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No true/*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset="channel", keep="last")


def build_max_null_from_perms(results_dir: str) -> np.ndarray:
    """Max-over-channels AUC for each permutation draw, from ``perms/perm-*.csv``."""
    import glob
    files = sorted(glob.glob(os.path.join(results_dir, "perms", "perm-*.csv")))
    if not files:
        raise FileNotFoundError(f"No perms/perm-*.csv in {results_dir}")
    return np.asarray(
        [float(np.nanmax(pd.read_csv(f)["auc"].to_numpy(dtype=float))) for f in files],
        dtype=float,
    )


def merge_spatial_maxstat(results_dir: str, alpha: float = 0.05,
                          montage: str = DEFAULT_MONTAGE, title: str = "Decoding AUC",
                          cmap: str = "viridis",
                          vlim: tuple | None = (0.5, None)) -> "pd.DataFrame":
    """
    Combine true per-channel AUCs + permutation max-null into FWER metrics + topomaps.

    Reads ``true/`` and ``perms/`` under ``results_dir``, builds the max-over-channels
    null, computes per-channel family-wise (FWER) p-values via the max-statistic test,
    writes ``per_channel_metrics.csv``, and renders ``topomap_auc.png`` (+ FWER-masked
    ``topomap_sig.png``).

    Parameters
    ----------
    results_dir : str
        ``…/SpatialDecoding/{LOSO|WithinSubject}/{contrast}/{family}/{model}``.
    alpha : float
        FWER level.
    montage : str
        MNE standard montage name.
    title : str
        Topomap title prefix.
    cmap : str
        Topomap colormap (default ``viridis``).
    vlim : tuple or None
        (vmin, vmax) AUC colour limits (default ``(0.5, None)``: vmin=0.5 fixed, vmax
        auto-set to each map's own maximum). Values below vmin are clamped to the bottom
        colour, so every chance-level (≤0.5) electrode looks the same and the 0.5→max
        range carries all the contrast.

    Returns
    -------
    pd.DataFrame
        The per-channel metrics table.
    """
    true_df = load_true_per_channel(results_dir)
    max_null = build_max_null_from_perms(results_dir)

    # FWER test uses the single-pass AUC (matched to the single-pass permutation null);
    # the displayed/topomap AUC is the n_runs-averaged mean_auc. Fall back to mean_auc
    # if a run did not record auc_single (older outputs).
    test_col = "auc_single" if "auc_single" in true_df.columns else "mean_auc"
    test_stat = dict(zip(true_df["channel"], true_df[test_col]))

    fwer = maxstat_pvalues(test_stat, max_null, alpha=alpha)  # mean_auc here == test stat
    fwer = fwer.drop(columns=["mean_auc"])  # keep perm_p / sig / fwer_threshold only

    carry = [c for c in ("mean_auc", "auc_single", "n_features", "std_auc")
             if c in true_df.columns]
    metrics = true_df[["channel", *carry]].merge(fwer, on="channel", how="left")
    metrics = metrics.sort_values("channel").reset_index(drop=True)
    metrics.to_csv(os.path.join(results_dir, "per_channel_metrics.csv"), index=False)

    _vlim = tuple(vlim) if vlim is not None else None
    plot_channel_topomap(metrics, "mean_auc", os.path.join(results_dir, "topomap_auc.png"),
                         montage=montage, title=title, cmap=cmap, vlim=_vlim)
    plot_channel_topomap(metrics, "mean_auc", os.path.join(results_dir, "topomap_sig.png"),
                         montage=montage, mask_col="sig", cmap=cmap, vlim=_vlim,
                         title=f"{title} (FWER-significant electrodes marked)")
    return metrics


def build_info_from_channels(channels: list[str], montage: str = DEFAULT_MONTAGE):
    """
    Build an MNE Info with electrode positions for the given channels.

    Defaults to the project's real cap montage (``CACS-64_REF.bvef``, read with
    ``read_custom_montage``) so spatial topomaps match the CBPT topoplots exactly. A
    standard montage name (e.g. ``standard_1020``) is also accepted and built with
    ``make_standard_montage``.

    Parameters
    ----------
    channels : list of str
        EEG channel names (must exist in the montage).
    montage : str
        Path to a montage file (``.bvef``/``.elc``/…) or an MNE standard montage name.

    Returns
    -------
    mne.Info
        Info object with the montage applied, ready for ``plot_topomap``.
    """
    import mne

    montage_arg = montage
    # Resolve a relative montage path against the repo root.
    if not os.path.isabs(montage_arg) and not str(montage_arg).startswith("standard"):
        cand = Path(__file__).resolve().parents[2] / montage_arg
        if cand.exists():
            montage_arg = str(cand)

    if os.path.exists(montage_arg):
        mont = mne.channels.read_custom_montage(montage_arg)
    else:
        mont = mne.channels.make_standard_montage(montage_arg)

    info = mne.create_info(ch_names=list(channels), sfreq=1.0, ch_types="eeg")
    info.set_montage(mont, match_case=False, on_missing="ignore")
    return info


def plot_channel_topomap(
    metrics: pd.DataFrame,
    value_col: str,
    out_path: str,
    montage: str = DEFAULT_MONTAGE,
    mask_col: str | None = None,
    title: str = "",
    cmap: str = "viridis",
    vlim: tuple | None = (0.5, None),
) -> None:
    """
    Render a scalp topomap of a per-channel metric using the EXACT plotting code of the
    project's CBPT topoplots (``Statistics/plot_results.py::plot_cluster_topomap``):
    same figure size, head-extrapolated cubic interpolation, 6 contours, circular clip,
    filled-black-dot significance markers, and colorbar. Only the colormap (viridis),
    the value scale (vmin/vmax), and the colorbar label differ from the CBPT default.

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
        Boolean column flagging significant electrodes (filled black dots).
    title : str
        Figure title.
    cmap : str
        Colormap (default ``viridis``).
    vlim : tuple or None
        (vmin, vmax); ``vmax=None`` -> data max. Default ``(0.5, None)``.
    """
    channels = metrics["channel"].tolist()
    info = build_info_from_channels(channels, montage=montage)
    values = metrics[value_col].to_numpy(dtype=float)

    # Express the FWER-significant electrodes as a single "cluster" so the CBPT plotter
    # marks them with filled black dots.
    if mask_col and mask_col in metrics:
        sig_idx = np.where(metrics[mask_col].to_numpy(dtype=bool))[0]
        clusters = [sig_idx] if sig_idx.size else []
        cluster_p_values = np.array([0.0]) if sig_idx.size else np.array([])
    else:
        clusters, cluster_p_values = [], np.array([])

    vmin = 0.5 if vlim is None else (vlim[0] if vlim[0] is not None else 0.5)
    vmax = (vlim[1] if (vlim is not None and vlim[1] is not None) else float(np.nanmax(values)))

    _plot_cbpt_style_topomap(
        values, clusters, cluster_p_values, info,
        alpha=0.05, title=title, vmin=vmin, vmax=vmax, cmap=cmap,
        save_path=out_path, cbar_label=value_col, footer=None,
    )


def _draw_cbpt_topomap(ax, values: np.ndarray, info, mask: np.ndarray | None,
                       vmin: float, vmax: float, cmap: str, markersize: int = 6):
    """
    Draw ONE CBPT-style topomap into ``ax`` and return the image handle.

    This is the exact per-axes rendering of ``Statistics/plot_results.py`` (sensors off,
    6 contours, head extrapolation, cubic interpolation, filled-black-dot mask, circular
    clip). Both the single-map and the multi-panel figures call this, so they render
    identically.
    """
    import mne
    from matplotlib.patches import Circle

    v = np.copy(values)
    v[np.isnan(v)] = 0
    im, _ = mne.viz.plot_topomap(
        v, info, axes=ax, show=False, cmap=cmap, vlim=(vmin, vmax),
        sensors=False, mask=mask,
        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                         linewidth=0, markersize=markersize),
        contours=6, ch_type='eeg', sphere=None, outlines='head',
        extrapolate='head', image_interp='cubic', border='mean', res=128,
    )
    circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor='none', edgecolor='none')
    im.set_clip_path(circle)
    return im


def _plot_cbpt_style_topomap(
    t_stats: np.ndarray,
    clusters: list,
    cluster_p_values: np.ndarray,
    info,
    alpha: float = 0.05,
    title: str = "Spatial Cluster Permutation Test",
    vmin=None,
    vmax=None,
    cmap: str = "RdBu_r",
    save_path: str | None = None,
    dpi: int = 300,
    cluster_p_values_uncorrected=None,
    cbar_label: str = "T-statistic",
    footer: str | None = "auto",
):
    """
    Single CBPT-style topomap (figure + colorbar), built on ``_draw_cbpt_topomap`` —
    the exact per-axes rendering of ``Statistics/plot_results.py::plot_cluster_topomap``.
    Additions vs the original: ``cbar_label`` and ``footer`` (None omits the caption).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    mask_corrected = np.zeros(len(t_stats), dtype=bool)
    for cluster, pval in zip(clusters, cluster_p_values):
        if pval < alpha:
            mask_corrected[cluster] = True

    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(t_stats))
        if np.isnan(abs_max) or abs_max == 0:
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max

    fig, ax = plt.subplots(figsize=(8, 6))
    im = _draw_cbpt_topomap(ax, t_stats, info, mask_corrected, vmin, vmax, cmap)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if footer == "auto":
        n_sig = int(np.sum(np.asarray(cluster_p_values) < alpha))
        fig.text(0.5, 0.02, f'Clusters: {n_sig}/{len(clusters)} significant (α = {alpha})',
                 ha='center', fontsize=9)
    elif footer:
        fig.text(0.5, 0.02, footer, ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_combined_topomap_panel(
    per_dimension_metrics: dict,
    value_col: str,
    out_path: str,
    montage: str = DEFAULT_MONTAGE,
    mask_col: str | None = None,
    cmap: str = "viridis",
    vlim: tuple | None = (0.5, None),
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

    dims = list(per_dimension_metrics.keys())
    # Shared colour scale across panels so dimensions are directly comparable. With
    # vmax=None, use the global maximum across all dimensions (vmin stays 0.5).
    vmin = 0.5 if vlim is None else (vlim[0] if vlim[0] is not None else 0.5)
    if vlim is not None and vlim[1] is not None:
        vmax = vlim[1]
    else:
        vmax = max(float(m[value_col].max()) for m in per_dimension_metrics.values())

    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims), 3.6))
    if len(dims) == 1:
        axes = [axes]
    im = None
    for ax, dim in zip(axes, dims):
        m = per_dimension_metrics[dim]
        info = build_info_from_channels(m["channel"].tolist(), montage=montage)
        mask = m[mask_col].to_numpy(dtype=bool) if mask_col else None
        # Same per-axes renderer as the single-map figures (identical style).
        im = _draw_cbpt_topomap(ax, m[value_col].to_numpy(dtype=float), info, mask,
                                vmin, vmax, cmap, markersize=5)
        ax.set_title(dim, fontsize=11, fontweight="bold")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, label=value_col)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
