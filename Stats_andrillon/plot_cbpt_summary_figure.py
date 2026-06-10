"""
CBPT Summary Figure — Paper 2 EEG
==================================
Panel A: Topomaps of representative markers (rows=families, cols=dimensions).
         Significant electrodes shown in red/blue; non-significant in grey.
Panel B: Heatmap (all markers × dimensions). Red=positive cluster,
         blue=negative cluster, white=non-significant.

Usage:
    conda activate plots
    python3.12 Stats_andrillon/plot_cbpt_summary_figure.py
"""

import os
import re
import sys
import csv
import glob
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

import mne

# ===== Paths =====

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'andrillon_cluster')
CHAN_ORDER    = os.path.join(PROJECT_ROOT, 'results', 'lmm_cluster', 'channel_order.txt')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'results', 'figures')
CONFIG_PATH  = os.path.join(PROJECT_ROOT, 'Stats_andrillon', 'config_andrillon.yaml')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_active_marker_lookup(config_path: str) -> dict:
    """Return {marker_name: epoch_type} for markers in MARKER_GROUPS / PANEL_A_MARKERS.

    Resolves selected_markers × feature_families (substring match, same logic as
    get_marker_list) + derived_ratios (exact match) without scanning the filesystem.
    Markers not covered by any active family are absent from the returned dict,
    so callers can use `.get(name)` to test whether a marker is active.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ff = config.get('feature_families', {})
    sm = config.get('selected_markers', {})
    dr = config.get('derived_ratios', {})

    # Build fragment → epoch_type from selected_markers × feature_families.
    frag_to_type: dict = {}
    for epoch_type, fam_list in sm.items():
        if not isinstance(fam_list, list):
            continue
        for fam in fam_list:
            for frag in (ff.get(fam) or []):
                frag_to_type[frag] = epoch_type
    # Derived ratios use exact names.
    for name, spec in dr.items():
        frag_to_type[name] = spec['epoch_type']

    # Resolve full marker names used in MARKER_GROUPS / PANEL_A_MARKERS via
    # substring match (mirrors get_marker_list behaviour).
    all_plot_names = (
        [mname for mname, _ in PANEL_A_MARKERS]
        + [mname for _, names in MARKER_GROUPS for mname in names]
    )
    lookup: dict = {}
    for name in all_plot_names:
        for frag, epoch_type in frag_to_type.items():
            if frag in name:
                lookup[name] = epoch_type
                break
    return lookup


# ===== Configuration =====

DIMS = [
    ('onoff_valence_selfother_time_time_on_task_confidence__target_onoff',
     'on/off'),
    ('onoff_valence_selfother_time_time_on_task_confidence__target_selfother',
     'self/other'),
    ('onoff_valence_selfother_time_time_on_task_confidence__target_valence',
     'valence'),
    ('onoff_valence_selfother_time_time_on_task_confidence__target_time',
     'time'),
    ('onoff_x_confidence_valence_selfother_time_time_on_task__target_onoff_confidence',
     'on/off × conf'),
    ('valence_x_confidence_onoff_selfother_time_time_on_task__target_valence_confidence',
     'valence × conf'),
]

# Representative markers for Panel A — (marker_name, row_label).
# Epoch type is resolved at runtime from the config via ACTIVE_MARKER_LOOKUP.
PANEL_A_MARKERS = [
    ('P3b',             'Attentional ERP\n(P3b)'),
    ('PE_gamma',        'Complexity\n(PE γ)'),
    ('psd_bands_gamma', 'Broadband power\n(PSD γ)'),
    ('psd_bands_alpha', 'Sleep oscillations\n(α power)'),
]

# Marker groups for Panel B — (group_label, [marker_names]).
# Epoch types are NOT hardcoded here; they are resolved from the config at
# runtime via ACTIVE_MARKER_LOOKUP.  Markers absent from the config's
# selected_markers are automatically excluded from the figure.
MARKER_GROUPS = [
    ('ERPs', [
        'N1', 'P1', 'P3a', 'P3b',
    ]),
    ('Complexity', [
        'kolmogorov_complexity',
        'PE_alpha', 'PE_beta', 'PE_gamma', 'PE_theta',
        'per_channel_msf_psdsummary',
        'per_channel_sef90_psdsummary', 'per_channel_sef95_psdsummary',
    ]),
    ('Spectral power', [
        'psd_bands_alpha', 'psd_bands_beta', 'psd_bands_gamma',
        'psd_bands_delta', 'psd_bands_theta', 'psd_bands_jota',
        'psd_bands_theta_beta_ratio', 'psd_bands_alpha_theta_ratio',
        'psd_relative_alpha', 'psd_relative_beta', 'psd_relative_gamma',
        'psd_relative_delta', 'psd_relative_theta', 'psd_relative_jota',
        'psd_relative_theta_beta_ratio', 'psd_relative_alpha_theta_ratio',
    ]),
    ('Connectivity', [
        'wsmi_alpha', 'wsmi_beta', 'wsmi_gamma', 'wsmi_theta',
    ]),
    ('Sleep events', [
        'slowwaves_Density', 'slowwaves_Duration',
        'slowwaves_Frequency', 'slowwaves_PTP', 'slowwaves_Slope',
        'spindles_Amplitude', 'spindles_Density',
    ]),
]

# Pretty labels for Panel B y-axis
MARKER_LABELS = {
    'kolmogorov_complexity': 'KoC',
    'per_channel_msf_psdsummary': 'MSF',
    'per_channel_sef90_psdsummary': 'SEF90',
    'per_channel_sef95_psdsummary': 'SEF95',
    'psd_bands_alpha': 'α power',
    'psd_bands_beta': 'β power',
    'psd_bands_gamma': 'γ power',
    'psd_bands_delta': 'δ power',
    'psd_bands_theta': 'θ power',
    'psd_bands_jota': 'ι power',
    'psd_bands_alpha_theta_ratio': 'α/θ ratio',
    'psd_bands_theta_beta_ratio': 'θ/β ratio',
    'psd_relative_alpha': 'rel. α',
    'psd_relative_beta': 'rel. β',
    'psd_relative_gamma': 'rel. γ',
    'psd_relative_delta': 'rel. δ',
    'psd_relative_theta': 'rel. θ',
    'psd_relative_jota': 'rel. ι',
    'psd_relative_alpha_theta_ratio': 'rel. α/θ ratio',
    'psd_relative_theta_beta_ratio': 'rel. θ/β ratio',
    'slowwaves_Density': 'SW density',
    'slowwaves_Duration': 'SW duration',
    'slowwaves_Frequency': 'SW frequency',
    'slowwaves_PTP': 'SW PTP',
    'slowwaves_Slope': 'SW slope',
    'spindles_Amplitude': 'spindle amp.',
    'spindles_Density': 'spindle density',
    'wsmi_alpha': 'wSMI α',
    'wsmi_beta': 'wSMI β',
    'wsmi_gamma': 'wSMI γ',
    'wsmi_theta': 'wSMI θ',
}


# ===== Helpers =====

def load_channel_info():
    """Build MNE Info with standard_1020 positions for our 64 channels."""
    raw = open(CHAN_ORDER).readlines()
    ch_names = [l.split(': ')[1].strip() for l in raw if re.match(r'\s+\d+:', l)]
    info = mne.create_info(ch_names, sfreq=1., ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='raise', verbose=False)
    return info, ch_names


def load_clusters(dim_folder: str, mtype: str, mname: str) -> list[tuple]:
    """Return list of (sign, n_electrodes, p_value, electrodes) for significant clusters."""
    path = os.path.join(
        RESULTS_DIR, dim_folder,
        f'{mtype}_{mname}',
        f'{mtype}_{mname}_clusters.csv',
    )
    if not os.path.exists(path):
        return []
    clusters = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if float(row['p_value']) < 0.05:
                sign = 1 if row['cluster_type'] == 'positive' else -1
                n    = int(row['n_electrodes'])
                p    = float(row['p_value'])
                elec = [e.strip() for e in row['electrodes'].split(',')]
                clusters.append((sign, n, p, elec))
    return clusters


def clusters_to_topo_values(clusters: list, ch_names: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert cluster list to per-channel values and significance mask.

    Returns
    -------
    values : ndarray, shape (n_channels,)
        +1 positive cluster, -1 negative cluster, 0 non-significant.
    sig_mask : ndarray of bool, shape (n_channels,)
        True where electrode belongs to any significant cluster.
    """
    values   = np.zeros(len(ch_names))
    sig_mask = np.zeros(len(ch_names), dtype=bool)
    ch_idx   = {ch: i for i, ch in enumerate(ch_names)}

    for sign, _, _, elecs in clusters:
        for e in elecs:
            if e in ch_idx:
                values[ch_idx[e]]   = sign
                sig_mask[ch_idx[e]] = True
    return values, sig_mask


def cluster_heatmap_value(clusters: list) -> float:
    """Signed log(n_electrodes) of dominant cluster for heatmap colour."""
    if not clusters:
        return 0.0
    dominant = max(clusters, key=lambda x: x[1])
    sign     = dominant[0]
    total_n  = sum(c[1] for c in clusters)
    return sign * np.log1p(total_n)


# ===== Panel A =====

def draw_panel_a(fig, gs, info, ch_names):
    """Topomaps: rows = marker families, cols = 4 main dimensions."""
    dims_subset = DIMS[:4]

    for ri, (mname, family_label) in enumerate(PANEL_A_MARKERS):
        mtype = ACTIVE_MARKER_LOOKUP.get(mname, 'evoked')
        for ci, (dim_folder, dim_label) in enumerate(dims_subset):
            ax = fig.add_subplot(gs[ri, ci])

            clusters = load_clusters(dim_folder, mtype, mname)
            values, sig_mask = clusters_to_topo_values(clusters, ch_names)

            # If nothing significant, show flat grey map
            if not np.any(sig_mask):
                values_plot = np.zeros(len(ch_names))
                mask_params = None
            else:
                values_plot = values.astype(float)
                mask_params = dict(
                    marker='o',
                    markerfacecolor='none',
                    markeredgecolor='k',
                    markersize=4,
                    linewidth=1.2,
                )

            mne.viz.plot_topomap(
                values_plot,
                info,
                axes=ax,
                cmap='RdBu_r',
                vlim=(-1, 1),
                sensors=False,
                contours=0,
                mask=sig_mask if np.any(sig_mask) else None,
                mask_params=mask_params,
                show=False,
            )

            # Significance annotation
            if clusters:
                n_sig = sum(c[1] for c in clusters)
                dom   = max(clusters, key=lambda x: x[1])
                color = '#bb0000' if dom[0] > 0 else '#0022bb'
                ax.text(0.5, -0.08, f'n={n_sig}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=6.5,
                        color=color, fontweight='bold')
            else:
                ax.text(0.5, -0.08, 'n.s.', transform=ax.transAxes,
                        ha='center', va='top', fontsize=6.5, color='#888888')

            # Column headers (top row only)
            if ri == 0:
                ax.set_title(dim_label, fontsize=9, fontweight='bold', pad=6)

            # Row labels (left column only) — fake via text outside axes
            if ci == 0:
                ax.text(-0.18, 0.5, family_label, transform=ax.transAxes,
                        ha='right', va='center', fontsize=8,
                        rotation=0, multialignment='right')


# ===== Panel B =====

def draw_panel_b(ax, ch_names):
    """Heatmap: all markers with ≥1 significant result × all 6 dimensions."""

    # Flatten all markers preserving group order.
    # Only include markers that are active in the config (present in lookup).
    all_markers   = []  # list of (epoch_type, marker_name)
    group_slices  = []  # (start, end, label)
    for group_label, marker_names in MARKER_GROUPS:
        start = len(all_markers)
        for mname in marker_names:
            mtype = ACTIVE_MARKER_LOOKUP.get(mname)
            if mtype is not None:
                all_markers.append((mtype, mname))
        if len(all_markers) > start:
            group_slices.append((start, len(all_markers), group_label))

    n_rows = len(all_markers)
    n_cols = len(DIMS)

    matrix   = np.zeros((n_rows, n_cols))
    has_any  = np.zeros(n_rows, dtype=bool)

    for ci, (dim_folder, _) in enumerate(DIMS):
        for ri, (mtype, mname) in enumerate(all_markers):
            clusters = load_clusters(dim_folder, mtype, mname)
            val = cluster_heatmap_value(clusters)
            matrix[ri, ci] = val
            if clusters:
                has_any[ri] = True

    # Keep only rows with at least one significant result
    keep          = np.where(has_any)[0]
    matrix_filt   = matrix[keep]
    markers_filt  = [all_markers[i] for i in keep]
    idx_map       = {old: new for new, old in enumerate(keep)}

    # Recompute group slices for filtered rows
    filt_groups = []
    for start, end, label in group_slices:
        rows_in_group = [idx_map[i] for i in range(start, end) if i in idx_map]
        if rows_in_group:
            filt_groups.append((min(rows_in_group), max(rows_in_group) + 1, label))

    # Draw heatmap
    vmax = np.max(np.abs(matrix_filt[matrix_filt != 0])) if np.any(matrix_filt != 0) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(matrix_filt, cmap='RdBu_r', norm=norm,
                     aspect='auto', interpolation='none')

    # Grey out non-significant cells
    grey_mask = np.ma.masked_where(matrix_filt != 0, np.ones_like(matrix_filt))
    ax.imshow(grey_mask, cmap='Greys', vmin=0, vmax=1,
              aspect='auto', interpolation='none', alpha=0.45)

    # Y-axis labels — prefix with epoch type when it differs from evoked
    def fmt_label(mtype, mname):
        base = MARKER_LABELS.get(mname, mname.replace('_', ' '))
        return f'{mtype}: {base}' if mtype != 'evoked' else base

    ax.set_yticks(range(len(markers_filt)))
    ax.set_yticklabels(
        [fmt_label(mt, mn) for mt, mn in markers_filt],
        fontsize=7.5,
    )

    # X-axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([d[1] for d in DIMS], rotation=30, ha='right', fontsize=9)
    ax.xaxis.set_tick_params(length=0)

    n_filt = len(markers_filt)

    # Group separators (horizontal lines across full width)
    for start, end, label in filt_groups:
        if start > 0:
            ax.axhline(start - 0.5, color='#333333', linewidth=1.0, zorder=3)

    # Group labels: placed to the right of the axes using axes-fraction coords
    # x=1.02 puts them just outside the right edge; y is converted from data coords
    for start, end, label in filt_groups:
        mid      = (start + end - 1) / 2
        y_frac   = 1.0 - (mid + 0.5) / n_filt          # axes fraction (top=1, bottom=0)
        ax.text(1.02, y_frac, label.replace('\n', ' '),
                va='center', ha='left', fontsize=8, color='#222222',
                transform=ax.transAxes, clip_on=False,
                fontweight='bold')

    # Colorbar — placed as a separate inset far to the right so it never overlaps labels
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax  = inset_axes(ax, width='2%', height='30%', loc='lower right',
                      bbox_to_anchor=(1.22, 0.0, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('log(n elec.) × direction', fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title('B — All EEG markers across MW dimensions',
                 fontsize=10, fontweight='bold', loc='left', pad=8)


# ===== Main =====

def main():
    global ACTIVE_MARKER_LOOKUP
    ACTIVE_MARKER_LOOKUP = _build_active_marker_lookup(CONFIG_PATH)
    print(f"Active markers from config: {len(ACTIVE_MARKER_LOOKUP)} entries")

    info, ch_names = load_channel_info()

    # Figure layout
    fig = plt.figure(figsize=(13, 26))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Neural signatures of mind-wandering dimensions\n(Andrillon CBPT, LMM t-values)',
        fontsize=12, fontweight='bold', y=0.995,
    )

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 2.8],
        hspace=0.08,
    )

    # --- Panel A ---
    gs_a = gridspec.GridSpecFromSubplotSpec(
        len(PANEL_A_MARKERS), 4,
        subplot_spec=outer[0],
        hspace=0.55, wspace=0.05,
    )
    fig.text(0.01, 0.76, 'A', fontsize=14, fontweight='bold', va='top')
    draw_panel_a(fig, gs_a, info, ch_names)

    # --- Panel B ---
    ax_b = fig.add_subplot(outer[1])
    draw_panel_b(ax_b, ch_names)
    fig.text(0.01, 0.46, 'B', fontsize=14, fontweight='bold', va='top')

    # Right margin: leave room for Panel B group labels + colorbar
    fig.subplots_adjust(right=0.72)

    # Save
    for fmt in ('pdf', 'png'):
        out_path = os.path.join(OUTPUT_DIR, f'cbpt_summary_figure.{fmt}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out_path}')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
