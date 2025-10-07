import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.gridspec import GridSpec
import os
import sys

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (script is in the project root)
project_root = os.path.abspath(script_dir)


def plot_condition_difference_topomap(df, marker, condition_col,
                                      condition_high='high', 
                                      condition_low='low',
                                      montage_name='standard_1020',
                                      save_to=None, figsize=(15, 5)):
    """
    Plot topographic maps showing condition high, condition low, and their 
    difference.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Aggregated marker data with columns: subject_id, task, marker, 
        channel, condition columns, mean, std, count
    marker : str
        The marker to plot (e.g., 'a', 'a_n', 'wSMI_1')
    condition_col : str
        The condition column to compare (e.g., 'onoff_label', 'valence_label')
    condition_high : str
        Label for high condition (default: 'high')
    condition_low : str
        Label for low condition (default: 'low')
    montage_name : str
        MNE montage name for electrode positions
    save_to : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Filter data for the specific marker
    marker_data = df[df['marker'] == marker].copy()
    
    if marker_data.empty:
        print(f"No data found for marker '{marker}'")
        return None
    
    # Check if condition column exists
    if condition_col not in marker_data.columns:
        print(f"Condition column '{condition_col}' not found in data")
        print(f"Available columns: {list(marker_data.columns)}")
        return None
    
    # Group by channel and condition, calculate mean across subjects/tasks
    grouped = marker_data.groupby(['channel', condition_col])['mean'].mean()
    grouped = grouped.reset_index()
    
    # Separate high and low conditions
    high_data = grouped[grouped[condition_col] == condition_high]
    low_data = grouped[grouped[condition_col] == condition_low]
    
    if high_data.empty or low_data.empty:
        print(f"No data found for conditions '{condition_high}' or "
              f"'{condition_low}'")
        print(f"Available conditions: {marker_data[condition_col].unique()}")
        return None
    
    # Merge high and low data on channel
    merged = pd.merge(high_data, low_data, on='channel',
                      suffixes=('_high', '_low'))
    
    # Calculate difference (high - low)
    merged['difference'] = merged['mean_high'] - merged['mean_low']
    
    # Get channel positions from MNE montage
    montage = mne.channels.make_standard_montage(montage_name)
    
    # Filter channels that exist in the montage
    available_channels = [ch for ch in merged['channel']
                         if ch in montage.ch_names]
    merged_filtered = merged[merged['channel'].isin(available_channels)]
    
    if merged_filtered.empty:
        print(f"No channels found in montage '{montage_name}'")
        return None
    
    # Create MNE info object
    info = mne.create_info(ch_names=available_channels,
                          sfreq=250., ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    
    # Prepare data arrays
    ch_data = merged_filtered.set_index('channel')
    high_values = np.array([ch_data.loc[ch, 'mean_high']
                           for ch in available_channels])
    low_values = np.array([ch_data.loc[ch, 'mean_low']
                          for ch in available_channels])
    diff_values = np.array([ch_data.loc[ch, 'difference']
                           for ch in available_channels])
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[0.1, 1])
    
    # Add main title
    condition_name = condition_col.replace("_label", "").title()
    fig.suptitle(f'{marker} - {condition_name} Comparison',
                 fontsize=16, fontweight='bold')
    
    # Determine common scale for high and low conditions
    vmin_common = min(high_values.min(), low_values.min())
    vmax_common = max(high_values.max(), low_values.max())
    
    # Plot high condition
    ax1 = fig.add_subplot(gs[1, 0])
    im1, _ = mne.viz.plot_topomap(high_values, info, show=False, axes=ax1,
                                  cmap='viridis', contours=6, sensors=True,
                                  names=available_channels, outlines='head')
    ax1.set_title(f'{condition_high.title()} Condition')
    
    # Add text showing value range for high condition
    ax1.text(0.02, 0.98, f'Range: {high_values.min():.3f} to {high_values.max():.3f}',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar for high condition
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label(f'{marker} value')
    
    # Plot low condition  
    ax2 = fig.add_subplot(gs[1, 1])
    im2, _ = mne.viz.plot_topomap(low_values, info, show=False, axes=ax2,
                                  cmap='viridis', contours=6, sensors=True,
                                  names=available_channels, outlines='head')
    ax2.set_title(f'{condition_low.title()} Condition')
    
    # Add text showing value range for low condition
    ax2.text(0.02, 0.98, f'Range: {low_values.min():.3f} to {low_values.max():.3f}',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar for low condition
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label(f'{marker} value')
    
    # Plot difference
    ax3 = fig.add_subplot(gs[1, 2])
    # Use symmetric scale for difference plot
    diff_max = max(abs(diff_values.min()), abs(diff_values.max()))
    im3, _ = mne.viz.plot_topomap(diff_values, info, show=False, axes=ax3,
                                  cmap='RdBu_r', contours=6, sensors=True,
                                  names=available_channels, outlines='head')
    ax3.set_title(f'Difference ({condition_high} - {condition_low})')
    
    # Add text showing difference range
    ax3.text(0.02, 0.98, f'Range: {diff_values.min():.3f} to {diff_values.max():.3f}',
             transform=ax3.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar for difference plot
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, aspect=20)
    cbar3.set_label(f'{marker} difference\n({condition_high} - {condition_low})')
    
    # Add reference lines at zero (if the colorbar supports it)
    try:
        cbar3.ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    except:
        pass
    
    # Add text annotations next to the difference plot
    ax3.text(1.15, 0.8, 'Higher in\nON-task', transform=ax3.transAxes, 
             fontsize=8, ha='left', va='center', color='red')
    ax3.text(1.15, 0.2, 'Higher in\nOFF-task', transform=ax3.transAxes, 
             fontsize=8, ha='left', va='center', color='blue')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_to}")
    
    return fig


def main():
    """
    Main function to create difference topoplots for EEG markers.
    """
    # Define file paths - update these paths as needed
    csv_file_path = ('../results/aggregated_mne_markers/'
                     'aggregated_mne_markers_onoff_5trials_go_correct_subject.csv')
    
    # Handle paths to be relative to project root
    if csv_file_path.startswith('../') or not os.path.isabs(csv_file_path):
        csv_file_path = os.path.abspath(os.path.join(project_root, csv_file_path))
    
    print("Loading aggregated marker data...")
    try:
        # Load the aggregated data
        df = pd.read_csv(csv_file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Get all unique markers from the data
        available_markers = df['marker'].unique()
        print(f"Available markers: {available_markers}")
        
        # Filter out non-marker entries (like 'marker' header if it exists)
        markers = [m for m in available_markers if m != 'marker']
        
        available_conditions = [col for col in df.columns 
                               if col.endswith('_label')]
        print(f"Available conditions: {available_conditions}")
        
        # Check unique values in onoff_label column
        if 'onoff_label' in df.columns:
            print(f"Unique values in onoff_label: {df['onoff_label'].unique()}")
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please check the file path and ensure aggregated data exists.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create output directory
    output_dir = './results/topoplots_differences'
    if output_dir.startswith('./') or not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, output_dir.lstrip('./')))
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define the condition to analyze (on vs off-task)
    condition_col = 'onoff_label'
    condition_high = 'high'  # on-task
    condition_low = 'low'    # off-task
    
    # Plot difference topomaps for each marker
    successful_plots = 0
    failed_plots = 0
    
    print(f"\nProcessing {len(markers)} markers for {condition_col} comparison...")
    print(f"Comparing {condition_high} (on-task) vs {condition_low} (off-task)")
    print("-" * 60)
    
    for i, marker in enumerate(markers, 1):
        print(f"[{i}/{len(markers)}] Processing marker: {marker}")
        
        # Create the difference topoplot
        save_path = os.path.join(output_dir,
                               f"{marker}_{condition_col}_difference_"
                               f"topomap.png")
        
        fig = plot_condition_difference_topomap(
            df=df,
            marker=marker,
            condition_col=condition_col,
            condition_high=condition_high,
            condition_low=condition_low,
            save_to=save_path,
            figsize=(15, 5)
        )
        
        if fig is not None:
            successful_plots += 1
            plt.close(fig)  # Close to save memory
            print(f"  ✓ Successfully created plot for {marker}")
        else:
            failed_plots += 1
            print(f"  ✗ Failed to create plot for {marker}")
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Successfully created: {successful_plots} plots")
    print(f"  Failed: {failed_plots} plots")
    print(f"  All plots saved to: {output_dir}")
    
    if successful_plots > 0:
        print(f"\n🎉 Generated topoplots comparing on-task vs off-task for {successful_plots} markers!")
    else:
        print(f"\n⚠️  No plots were successfully generated. Please check the data and parameters.")


if __name__ == "__main__":
    main() 