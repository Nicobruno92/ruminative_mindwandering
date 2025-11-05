"""
Diagnose adjacency matrix to understand why non-adjacent channels are clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.sparse import csr_matrix, coo_array, issparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics')
from cluster_test import get_channel_adjacency


def diagnose_adjacency(adjacency, ch_names, montage_path, output_dir="adjacency_diagnostics"):
    """
    Comprehensive adjacency matrix diagnostics with visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert sparse to dense
    if issparse(adjacency):
        adj_dense = adjacency.toarray()
    else:
        adj_dense = adjacency.copy()
    
    # 1. Plot the adjacency matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(adj_dense, cmap='RdBu_r', aspect='auto')
    ax.set_title('Adjacency Matrix\n(Red = connected, Blue = not connected)')
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Channel index')
    
    # Add channel names for selected indices
    n_channels = len(ch_names)
    if n_channels <= 64:
        tick_positions = range(0, n_channels, max(1, n_channels // 20))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([ch_names[i] for i in tick_positions], rotation=90, fontsize=8)
        ax.set_yticklabels([ch_names[i] for i in tick_positions], fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjacency_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved adjacency matrix plot to {output_dir}/adjacency_matrix.png")
    
    # 2. Connectivity statistics per channel
    connections_per_channel = np.sum(adj_dense > 0, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(connections_per_channel, bins=range(0, max(connections_per_channel) + 2), 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of neighbors')
    axes[0].set_ylabel('Number of channels')
    axes[0].set_title(f'Distribution of Connections\nMean: {np.mean(connections_per_channel):.1f}')
    axes[0].axvline(x=np.mean(connections_per_channel), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    # List most and least connected
    sorted_idx = np.argsort(connections_per_channel)
    axes[1].text(0.1, 0.9, "Most connected channels:", fontsize=12, fontweight='bold', 
                transform=axes[1].transAxes)
    for i, idx in enumerate(sorted_idx[-5:]):
        axes[1].text(0.1, 0.8 - i*0.08, f"{ch_names[idx]}: {connections_per_channel[idx]} neighbors", 
                    fontsize=10, transform=axes[1].transAxes)
    
    axes[1].text(0.1, 0.4, "Least connected channels:", fontsize=12, fontweight='bold',
                transform=axes[1].transAxes)
    for i, idx in enumerate(sorted_idx[:5]):
        axes[1].text(0.1, 0.3 - i*0.08, f"{ch_names[idx]}: {connections_per_channel[idx]} neighbors",
                    fontsize=10, transform=axes[1].transAxes)
    
    axes[1].axis('off')
    axes[1].set_title('Channel Connectivity Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connectivity_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved connectivity stats plot to {output_dir}/connectivity_stats.png")
    
    # 3. Print detailed report
    print("\n" + "="*60)
    print("ADJACENCY MATRIX DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Total channels: {len(ch_names)}")
    print(f"Total connections: {np.sum(adj_dense > 0)}")
    print(f"Connections per channel: {np.mean(connections_per_channel):.1f} ± {np.std(connections_per_channel):.1f}")
    print(f"Min connections: {np.min(connections_per_channel)} ({ch_names[np.argmin(connections_per_channel)]})")
    print(f"Max connections: {np.max(connections_per_channel)} ({ch_names[np.argmax(connections_per_channel)]})")
    
    # Check for problems
    isolated = np.where(connections_per_channel == 0)[0]
    if len(isolated) > 0:
        print(f"\n⚠ ISOLATED CHANNELS ({len(isolated)}):")
        for idx in isolated:
            print(f"  - {ch_names[idx]}")
    
    overconnected = np.where(connections_per_channel > 10)[0]
    if len(overconnected) > 0:
        print(f"\n⚠ OVERCONNECTED CHANNELS (>10 neighbors, {len(overconnected)}):")
        for idx in overconnected:
            print(f"  - {ch_names[idx]}: {connections_per_channel[idx]} neighbors")
    
    print("="*60)


# Load the adjacency matrix
montage_path = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Preprocessing_pipeline_new/CACS-64_withREF.bvef"

# Get all 64 channels
ch_names = ['AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 
            'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
            'FT10', 'FT7', 'FT8', 'FT9', 'Fp1', 'Fp2', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 
            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 
            'POz', 'Pz', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']

print("Loading adjacency matrix...")
adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, ch_names)

# Run diagnostics with visualizations
output_dir = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics/tests/adjacency_diagnostics"
diagnose_adjacency(adjacency, ch_names_ordered, montage_path, output_dir)

# Convert to dense for easier inspection
if hasattr(adjacency, 'toarray'):
    adj_dense = adjacency.toarray()
else:
    adj_dense = adjacency.copy()

print(f"\nAdjacency matrix shape: {adj_dense.shape}")

# Check specific problematic channels from Cluster 1
problem_channels = ['FT9', 'FT10', 'TP9', 'TP10', 'Iz', 'Oz', 'AF7', 'AF8', 'Fp1', 'Fp2']

print("\n" + "="*70)
print("CHECKING ADJACENCY OF PROBLEMATIC CHANNELS")
print("="*70)

for ch in problem_channels:
    if ch in ch_names_ordered:
        ch_idx = ch_names_ordered.index(ch)
        adjacent_indices = np.where(adj_dense[ch_idx, :] > 0)[0]
        adjacent_names = [ch_names_ordered[i] for i in adjacent_indices if i != ch_idx]
        
        print(f"\n{ch} (index {ch_idx}):")
        print(f"  Adjacent to: {adjacent_names}")
        print(f"  Number of neighbors: {len(adjacent_names)}")

# Now check if these peripheral channels form a connected component
print("\n" + "="*70)
print("CHECKING IF PERIPHERAL CHANNELS ARE CONNECTED")
print("="*70)

peripheral_negative = ['FT9', 'FT10', 'TP9', 'TP10', 'Iz', 'Oz', 'AF7', 'AF8', 'Fp1', 'O1', 'O2', 'PO8']
peripheral_indices = [ch_names_ordered.index(ch) for ch in peripheral_negative if ch in ch_names_ordered]

print(f"\nPeripheral channels: {[ch_names_ordered[i] for i in peripheral_indices]}")

# Check connectivity between them
print("\n--- Using SUBGRAPH approach (BUGGY) ---")
subgraph = adj_dense[np.ix_(peripheral_indices, peripheral_indices)]
n_comp_sub, labels_sub = connected_components(csr_matrix(subgraph), directed=False)
print(f"Number of components in subgraph: {n_comp_sub}")
for comp_idx in range(n_comp_sub):
    comp_channels = [ch_names_ordered[peripheral_indices[i]] for i in range(len(peripheral_indices)) if labels_sub[i] == comp_idx]
    print(f"  Component {comp_idx}: {comp_channels}")

print("\n--- Using MASKED ADJACENCY approach (FIXED) ---")
masked_adj = np.zeros_like(adj_dense)
masked_adj[np.ix_(peripheral_indices, peripheral_indices)] = adj_dense[np.ix_(peripheral_indices, peripheral_indices)]
n_comp_mask, labels_mask = connected_components(csr_matrix(masked_adj), directed=False)

peripheral_components = []
for comp_idx in range(n_comp_mask):
    comp_all_indices = np.where(labels_mask == comp_idx)[0]
    comp_peripheral = [ch_names_ordered[i] for i in comp_all_indices if i in peripheral_indices]
    if len(comp_peripheral) > 0:
        peripheral_components.append(comp_peripheral)

print(f"Number of components containing peripheral channels: {len(peripheral_components)}")
for i, comp_channels in enumerate(peripheral_components):
    print(f"  Component {i}: {comp_channels}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
