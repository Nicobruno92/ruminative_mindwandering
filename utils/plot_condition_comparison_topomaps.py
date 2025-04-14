#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
from plot_topomaps import plot_marker_topography


def parse_condition_string(cond_str):
    """
    Parse a condition string like 'onoff_label=high,valence_label=low' into a dict.
    """
    conds = {}
    if cond_str:
        for part in cond_str.split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                conds[k.strip()] = v.strip()
    return conds


def filter_df_by_conditions(df, cond_dict):
    """
    Filter DataFrame by a dict of {col: value}.
    """
    for col, val in cond_dict.items():
        if col not in df.columns:
            raise ValueError(f"Condition column '{col}' not found in data. Available columns: {list(df.columns)}")
        df = df[df[col] == val]
    return df


def get_mean_per_channel(df, marker, groupby_cols=['channel']):
    """
    For a given marker, return a Series of mean values per channel (averaged across subjects/tasks if present).
    """
    df_marker = df[df['marker'] == marker]
    if df_marker.empty:
        raise ValueError(f"No data for marker '{marker}' after filtering.")
    mean_per_ch = df_marker.groupby(groupby_cols)['mean'].mean()
    return mean_per_ch


def cond_dict_to_str(cond_dict):
    """
    Convert a condition dict to a string for filenames/titles.
    """
    return '_'.join([f"{k}-{v}" for k, v in cond_dict.items()])


def plot_condition_comparison_topomaps(
    csv_file,
    markers,
    cond1,
    cond2,
    output_dir='topomap_comparisons',
    montage='standard_1020',
    eeglab_style=True,
    figsize=(10, 8),
    verbose=True
):
    """
    Main plotting function. Can be called from another script.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    cond1_dict = parse_condition_string(cond1) if isinstance(cond1, str) else cond1
    cond2_dict = parse_condition_string(cond2) if isinstance(cond2, str) else cond2
    results = []
    for marker in markers:
        try:
            df1 = filter_df_by_conditions(df, {**cond1_dict, 'marker': marker})
            df2 = filter_df_by_conditions(df, {**cond2_dict, 'marker': marker})
        except Exception as e:
            if verbose:
                print(f"Error filtering for marker {marker}: {e}")
            continue
        if df1.empty or df2.empty:
            if verbose:
                print(f"Warning: No data for marker {marker} and one of the conditions. Skipping.")
            continue
        try:
            mean1 = get_mean_per_channel(df1, marker)
            mean2 = get_mean_per_channel(df2, marker)
        except Exception as e:
            if verbose:
                print(f"Error getting mean per channel for marker {marker}: {e}")
            continue
        common_channels = sorted(set(mean1.index) & set(mean2.index))
        if not common_channels:
            if verbose:
                print(f"Warning: No common channels for marker {marker}. Skipping.")
            continue
        mean1 = mean1.loc[common_channels]
        mean2 = mean2.loc[common_channels]
        diff = mean1 - mean2
        # Build DataFrames for plotting
        plot_df1 = pd.DataFrame({f'{marker}_' + ch: [val] for ch, val in zip(common_channels, mean1)}, index=[0])
        plot_df2 = pd.DataFrame({f'{marker}_' + ch: [val] for ch, val in zip(common_channels, mean2)}, index=[0])
        plot_df_diff = pd.DataFrame({f'{marker}_' + ch: [val] for ch, val in zip(common_channels, diff)}, index=[0])
        # Titles and filenames
        cond1_str = cond_dict_to_str(cond1_dict)
        cond2_str = cond_dict_to_str(cond2_dict)
        base = f"{marker}__{cond1_str}__vs__{cond2_str}"
        # Plot cond1
        fig1 = plot_marker_topography(
            plot_df1, marker,
            title=f"{marker} | {cond1_str.replace('_', ' ')}",
            montage_name=montage,
            use_eeglab_style=eeglab_style,
            figsize=figsize
        )
        fname1 = os.path.join(output_dir, f"{base}__cond1.png")
        fig1.savefig(fname1, dpi=300, bbox_inches='tight')
        # Plot cond2
        fig2 = plot_marker_topography(
            plot_df2, marker,
            title=f"{marker} | {cond2_str.replace('_', ' ')}",
            montage_name=montage,
            use_eeglab_style=eeglab_style,
            figsize=figsize
        )
        fname2 = os.path.join(output_dir, f"{base}__cond2.png")
        fig2.savefig(fname2, dpi=300, bbox_inches='tight')
        # Plot difference
        figd = plot_marker_topography(
            plot_df_diff, marker,
            title=f"{marker} | Difference: ({cond1_str}) - ({cond2_str})",
            montage_name=montage,
            use_eeglab_style=eeglab_style,
            figsize=figsize
        )
        fname_diff = os.path.join(output_dir, f"{base}__diff.png")
        figd.savefig(fname_diff, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved: {fname1}\nSaved: {fname2}\nSaved: {fname_diff}")
        results.append({
            'marker': marker,
            'cond1': cond1_dict,
            'cond2': cond2_dict,
            'files': [fname1, fname2, fname_diff]
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot topomaps comparing two conditions for each marker.")
    parser.add_argument('csv_file', help='Aggregated per-electrode CSV file')
    parser.add_argument('--markers', nargs='+', required=True, help='Markers to plot (e.g., a a_n wSMI_1)')
    parser.add_argument('--cond1', required=True, help="First condition set, e.g. 'onoff_label=high,valence_label=high'")
    parser.add_argument('--cond2', required=True, help="Second condition set, e.g. 'onoff_label=high,valence_label=low'")
    parser.add_argument('--output_dir', default='topomap_comparisons', help='Directory to save plots')
    parser.add_argument('--montage', default='standard_1020', help='Montage name for plotting')
    parser.add_argument('--no-eeglab-style', action='store_false', dest='eeglab_style', help='Do not use EEGLAB-style layout')
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 8), help='Figure size (width height)')
    args = parser.parse_args()
    plot_condition_comparison_topomaps(
        csv_file=args.csv_file,
        markers=args.markers,
        cond1=args.cond1,
        cond2=args.cond2,
        output_dir=args.output_dir,
        montage=args.montage,
        eeglab_style=args.eeglab_style,
        figsize=tuple(args.figsize),
        verbose=True
    )

if __name__ == '__main__':
    main() 