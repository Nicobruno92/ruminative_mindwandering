"""
Visualize adjacency matrix as a topographic plot showing channel connections.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import sys

sys.path.insert(0, '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics')
from cluster_test import get_channel_adjacency

# Load the adjacency matrix
montage_path = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Preprocessing_pipeline_new/CACS-64_withREF.bvef"

ch_names = ['AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 
            'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
            'FT10', 'FT7', 'FT8', 'FT9', 'Fp1', 'Fp2', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 
            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 
            'POz', 'Pz', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']

print("Loading adjacency matrix...")
adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, ch_names)

# Convert to dense
if hasattr(adjacency, 'toarray'):
    adj_dense = adjacency.toarray()
else:
    adj_dense = adjacency.copy()

# Load montage for channel positions
if montage_path.endswith('.bvef'):
    montage = mne.channels.read_custom_montage(montage_path)
else:
    montage = mne.channels.make_standard_montage(montage_path)

# Create info object
info = mne.create_info(ch_names=ch_names_ordered, sfreq=250, ch_types='eeg')
info.set_montage(montage, on_missing='ignore')

# Get channel positions
pos = np.array([info['chs'][i]['loc'][:3] for i in range(len(ch_names_ordered))])

# Create output directory
output_dir = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics/tests/adjacency_diagnostics"

# Plot 1: Topographic view with all connections
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot the head outline and nose
mne.viz.plot_sensors(info, kind='topomap', show_names=True, axes=ax, show=False)

# Get 2D positions for plotting
pos_2d = mne.channels.layout._find_topomap_coords(info, picks='eeg')

# Draw connections
for i in range(len(ch_names_ordered)):
    for j in range(i+1, len(ch_names_ordered)):
        if adj_dense[i, j] > 0:
            # Draw line between connected channels
            ax.plot([pos_2d[i, 0], pos_2d[j, 0]], 
                   [pos_2d[i, 1], pos_2d[j, 1]], 
                   'b-', alpha=0.3, linewidth=0.5)

ax.set_title('Channel Adjacency Network\n(Blue lines = adjacent channels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/adjacency_topomap_all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved full adjacency topomap to {output_dir}/adjacency_topomap_all.png")

# Plot 2: Highlight specific problematic channels
problem_channels = ['FT9', 'FT10', 'TP9', 'TP10', 'Iz', 'Oz', 'AF7', 'AF8', 'Fp1', 'O1', 'O2', 'PO8']
problem_indices = [ch_names_ordered.index(ch) for ch in problem_channels if ch in ch_names_ordered]

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
mne.viz.plot_sensors(info, kind='topomap', show_names=False, axes=ax, show=False)

# Draw all connections in light gray
for i in range(len(ch_names_ordered)):
    for j in range(i+1, len(ch_names_ordered)):
        if adj_dense[i, j] > 0:
            ax.plot([pos_2d[i, 0], pos_2d[j, 0]], 
                   [pos_2d[i, 1], pos_2d[j, 1]], 
                   'gray', alpha=0.1, linewidth=0.3)

# Highlight connections between problem channels
for i in problem_indices:
    for j in problem_indices:
        if i < j and adj_dense[i, j] > 0:
            ax.plot([pos_2d[i, 0], pos_2d[j, 0]], 
                   [pos_2d[i, 1], pos_2d[j, 1]], 
                   'r-', alpha=0.8, linewidth=2, zorder=10)

# Mark problem channels
for idx in problem_indices:
    ax.plot(pos_2d[idx, 0], pos_2d[idx, 1], 'ro', markersize=10, zorder=11)
    ax.text(pos_2d[idx, 0], pos_2d[idx, 1], f'  {ch_names_ordered[idx]}', 
           fontsize=8, fontweight='bold', color='red', zorder=12)

ax.set_title('Peripheral Channel Connections\n(Red = problematic channels forming ring around head)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/adjacency_topomap_peripheral.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved peripheral adjacency topomap to {output_dir}/adjacency_topomap_peripheral.png")

# Plot 3: Show connectivity per channel as a topomap
connections_per_channel = np.sum(adj_dense > 0, axis=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im, cn = mne.viz.plot_topomap(connections_per_channel, info, axes=ax, show=False, 
                               cmap='viridis', contours=6, names=ch_names_ordered)
ax.set_title('Number of Neighbors per Channel', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Number of neighbors')
plt.tight_layout()
plt.savefig(f'{output_dir}/adjacency_topomap_connectivity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved connectivity topomap to {output_dir}/adjacency_topomap_connectivity.png")

# Plot 4: Show the peripheral ring path
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
mne.viz.plot_sensors(info, kind='topomap', show_names=False, axes=ax, show=False)

# Draw all connections in very light gray
for i in range(len(ch_names_ordered)):
    for j in range(i+1, len(ch_names_ordered)):
        if adj_dense[i, j] > 0:
            ax.plot([pos_2d[i, 0], pos_2d[j, 0]], 
                   [pos_2d[i, 1], pos_2d[j, 1]], 
                   'gray', alpha=0.05, linewidth=0.3)

# Trace the ring path
ring_path = ['AF7', 'FT9', 'TP9', 'Iz', 'TP10', 'FT10', 'AF8']
ring_indices = [ch_names_ordered.index(ch) for ch in ring_path]

# Draw the ring path
for i in range(len(ring_indices) - 1):
    idx1, idx2 = ring_indices[i], ring_indices[i+1]
    ax.plot([pos_2d[idx1, 0], pos_2d[idx2, 0]], 
           [pos_2d[idx1, 1], pos_2d[idx2, 1]], 
           'r-', alpha=0.8, linewidth=3, zorder=10)
    ax.annotate('', xy=(pos_2d[idx2, 0], pos_2d[idx2, 1]), 
               xytext=(pos_2d[idx1, 0], pos_2d[idx1, 1]),
               arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=11)

# Mark ring channels
for idx in ring_indices:
    ax.plot(pos_2d[idx, 0], pos_2d[idx, 1], 'ro', markersize=12, zorder=12)
    ax.text(pos_2d[idx, 0], pos_2d[idx, 1], f'  {ch_names_ordered[idx]}', 
           fontsize=9, fontweight='bold', color='red', zorder=13)

ax.set_title('Peripheral Ring Path\n(Shows how scattered channels are actually connected)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/adjacency_topomap_ring.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved ring path topomap to {output_dir}/adjacency_topomap_ring.png")

print("\n" + "="*70)
print("TOPOGRAPHIC VISUALIZATIONS COMPLETE")
print("="*70)
print(f"Output directory: {output_dir}")
print("\nGenerated plots:")
print("  1. adjacency_topomap_all.png - All channel connections")
print("  2. adjacency_topomap_peripheral.png - Peripheral channels highlighted")
print("  3. adjacency_topomap_connectivity.png - Number of neighbors per channel")
print("  4. adjacency_topomap_ring.png - The peripheral ring path")
print("="*70)
