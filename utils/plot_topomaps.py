#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from matplotlib.gridspec import GridSpec


def get_marker_data(df, marker):
    """
    Extract data for a specific marker from the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing EEG data.
    marker : str
        The marker to extract (e.g., 'a', 'a_n', 'wSMI_1').
        
    Returns
    -------
    dict
        Dictionary with channel names as keys and values as values.
    """
    # Get all columns that start with the marker name
    marker_cols = [col for col in df.columns 
                   if col.startswith(marker + '_') and '_n_' not in col]
    
    # Extract the channel names from the column names
    ch_names = [col.split('_', 1)[1] if '_' in col else col 
                for col in marker_cols]
    
    # Use the first subject's data
    subject_data = df.iloc[0]
    
    # Create a dictionary mapping channel names to values
    ch_data = {}
    for col, ch in zip(marker_cols, ch_names):
        ch_data[ch] = subject_data[col]
    
    return ch_data


def plot_marker_topography(df, marker, title=None, subject_idx=0, 
                           montage_name='standard_1020', use_eeglab_style=True, 
                           save_to=None, figsize=(10, 8)):
    """
    Plot a topographic map for a specific marker.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing EEG data.
    marker : str
        The marker to plot (e.g., 'a', 'a_n', 'wSMI_1').
    title : str, optional
        The title for the plot.
    subject_idx : int, optional
        The index of the subject to use for plotting.
    montage_name : str, optional
        The name of the montage to use.
    use_eeglab_style : bool, optional
        Whether to use EEGLAB-style channel layout.
    save_to : str, optional
        The filename to save the plot to.
    figsize : tuple, optional
        The figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    # Get all columns that start with the marker name
    marker_cols = [col for col in df.columns if col.startswith(marker + '_')]
    
    # Skip subject_id and round columns
    if 'subject_id' in marker_cols:
        marker_cols.remove('subject_id')
    if 'round' in marker_cols:
        marker_cols.remove('round')
    
    # Extract channel names from column names
    ch_names = []
    for col in marker_cols:
        parts = col.split('_')
        # Handle different marker formats
        if len(parts) == 2:  # e.g., a_AFz
            ch_names.append(parts[1])
        elif len(parts) > 2:  # e.g., wSMI_1_AFz
            ch_names.append(parts[-1])
    
    # Use the specified subject's data
    subject_data = df.iloc[subject_idx]
    
    # Create MNE Info object with channel positions
    montage = mne.channels.make_standard_montage(montage_name)
    
    # Keep only unique channels from the montage
    unique_channels = []
    available_marker_cols = []
    seen_channels = set()
    
    for ch, col in zip(ch_names, marker_cols):
        if ch in montage.ch_names and ch not in seen_channels:
            unique_channels.append(ch)
            available_marker_cols.append(col)
            seen_channels.add(ch)
    
    # Create info with unique available channels
    info = mne.create_info(
        ch_names=unique_channels, 
        sfreq=250., 
        ch_types='eeg'
    )
    
    # Get the data for available channels
    data = np.array([subject_data[col] for col in available_marker_cols])
    
    # Set montage and allow missing channels
    info.set_montage(montage, on_missing='ignore')
    
    # Calculate sphere origin and radius for EEGLAB-style layout
    sphere = None
    if use_eeglab_style:
        # Channels used to calculate the sphere
        chs = ['Oz', 'Fpz', 'T7', 'T8']
        montage_head = mne.channels.make_standard_montage(montage_name)
        temp_info = mne.create_info(
            ch_names=montage_head.ch_names, 
            sfreq=250., 
            ch_types='eeg'
        )
        temp_info.set_montage(montage_head, on_missing='ignore')
        
        # Get positions of channels
        pos = np.zeros((len(chs), 3))
        for i, ch in enumerate(chs):
            if ch in temp_info.ch_names:
                idx = temp_info.ch_names.index(ch)
                pos[i] = temp_info['chs'][idx]['loc'][:3]
        
        # Calculate radius and sphere center
        radius = np.abs(pos[[2, 3], 0]).mean()  # from T7 and T8 x position
        x = pos[0, 0]  # x position of Oz
        y = pos[3, 1]  # y position of T8
        z = pos[:, 2].mean()  # average z position
        
        sphere = (x, y, z, radius)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[0.1, 1])
    
    # Add title at the top
    if title is None:
        title = f'Topographic Map for {marker}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create main axis for the topomap
    ax_topo = fig.add_subplot(gs[1, 0])
    
    # Create axis for the colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    
    # Plot the topomap with channel labels
    im, cn = mne.viz.plot_topomap(
        data, info, 
        show=False, sphere=sphere, axes=ax_topo, 
        cmap='viridis', contours=6,
        sensors=True,  # Show sensor positions
        names=unique_channels,  # Pass channel names directly
        outlines='head',
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(marker)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
    
    return fig


def plot_marker_comparison(df, markers, subject_idx=0, montage_name='standard_1020',
                           use_eeglab_style=False, save_to=None, figsize=(12, 8)):
    """
    Plot multiple markers side by side for comparison.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing EEG data.
    markers : list of str
        The markers to plot.
    subject_idx : int, optional
        The index of the subject to use for plotting.
    montage_name : str, optional
        The name of the montage to use.
    use_eeglab_style : bool, optional
        Whether to use EEGLAB-style channel layout.
    save_to : str, optional
        The filename to save the plot to.
    figsize : tuple, optional
        The figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    n_markers = len(markers)
    
    # Create figure with a grid of subplots
    fig, axes = plt.subplots(1, n_markers, figsize=figsize)
    if n_markers == 1:
        axes = [axes]
    
    # Calculate sphere parameters for EEGLAB-style layout
    sphere = None
    if use_eeglab_style:
        # Channels used to calculate the sphere
        chs = ['Oz', 'Fpz', 'T7', 'T8']
        montage_head = mne.channels.make_standard_montage(montage_name)
        temp_info = mne.create_info(
            ch_names=montage_head.ch_names, 
            sfreq=250., 
            ch_types='eeg'
        )
        temp_info.set_montage(montage_head, on_missing='ignore')
        
        # Get positions of channels
        pos = np.zeros((len(chs), 3))
        for i, ch in enumerate(chs):
            if ch in temp_info.ch_names:
                idx = temp_info.ch_names.index(ch)
                pos[i] = temp_info['chs'][idx]['loc'][:3]
        
        # Calculate radius and sphere center
        radius = np.abs(pos[[2, 3], 0]).mean()  # from T7 and T8 x position
        x = pos[0, 0]  # x position of Oz
        y = pos[3, 1]  # y position of T8
        z = pos[:, 2].mean()  # average z position
        
        sphere = (x, y, z, radius)
    
    # Plot each marker
    for i, marker in enumerate(markers):
        # Get marker columns and channels
        marker_cols = [col for col in df.columns if col.startswith(marker + '_')]
        
        # Skip subject_id and round
        if 'subject_id' in marker_cols:
            marker_cols.remove('subject_id')
        if 'round' in marker_cols:
            marker_cols.remove('round')
        
        # Extract channel names
        ch_names = []
        for col in marker_cols:
            parts = col.split('_')
            if len(parts) == 2:  # e.g., a_AFz
                ch_names.append(parts[1])
            elif len(parts) > 2:  # e.g., wSMI_1_AFz
                ch_names.append(parts[-1])
        
        # Filter available channels
        montage = mne.channels.make_standard_montage(montage_name)
        
        # Keep only unique channels from the montage
        unique_channels = []
        available_marker_cols = []
        seen_channels = set()
        
        for ch, col in zip(ch_names, marker_cols):
            if ch in montage.ch_names and ch not in seen_channels:
                unique_channels.append(ch)
                available_marker_cols.append(col)
                seen_channels.add(ch)
        
        # Create info
        info = mne.create_info(
            ch_names=unique_channels, 
            sfreq=250., 
            ch_types='eeg'
        )
        info.set_montage(montage, on_missing='ignore')
        
        # Get data for available channels
        subject_data = df.iloc[subject_idx]
        data = np.array([subject_data[col] for col in available_marker_cols])
        
        # Plot topomap with channel labels
        im, _ = mne.viz.plot_topomap(
            data, info, axes=axes[i], 
            show=False, sphere=sphere, 
            cmap='viridis', contours=6,
            sensors=True,  # Show sensor positions
            names=unique_channels,  # Pass channel names directly
            outlines='head',
        )
        
        # Add title
        axes[i].set_title(marker, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(
            im, ax=axes[i], orientation='vertical', 
            fraction=0.046, pad=0.04
        )
        cbar.set_label(marker)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot topographic maps of EEG markers.')
    parser.add_argument('csv_file', help='Path to the CSV file containing EEG data')
    parser.add_argument(
        '--markers', nargs='+', required=True, 
        help='Markers to plot (e.g., a, a_n, wSMI_1)'
    )
    parser.add_argument(
        '--subject', type=int, default=0, 
        help='Subject index to use (default: 0)'
    )
    parser.add_argument(
        '--montage', default='standard_1020', 
        help='Montage name to use (default: standard_1020)'
    )
    parser.add_argument(
        '--no-eeglab-style', action='store_false', dest='eeglab_style',
        help='Do not use EEGLAB-style channel layout'
    )
    parser.add_argument('--save', help='Save the plots to files (prefix will be used)')
    parser.add_argument(
        '--compare', action='store_true', 
        help='Compare markers side by side'
    )
    
    args = parser.parse_args()
    
    # Load the data
    df = pd.read_csv(args.csv_file)
    
    if args.compare and len(args.markers) > 1:
        # Compare markers side by side
        _ = plot_marker_comparison(
            df, args.markers, subject_idx=args.subject,
            montage_name=args.montage, 
            use_eeglab_style=args.eeglab_style,
            save_to=args.save + '_comparison.png' 
            if args.save else None
        )
        plt.show()
    else:
        # Plot each marker separately
        for marker in args.markers:
            _ = plot_marker_topography(
                df, marker, subject_idx=args.subject,
                montage_name=args.montage, 
                use_eeglab_style=args.eeglab_style,
                save_to=args.save + f'_{marker}.png' 
                if args.save else None
            )
            plt.show()


if __name__ == '__main__':
    main()