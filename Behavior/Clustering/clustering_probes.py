#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

df = pd.read_csv('../../results/Behavior/probe_data/probe_level_aggregated_data.csv')
df = df[df['onoff'] <=60]


#%%
# Cluster Analysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Prepare data for clustering
cluster_variables = ['valence', 'selfother', 'time']
cluster_data = df[cluster_variables].dropna()

# Standardize the data
# Scale by participant - group by participant and standardize within each participant
cluster_data_scaled = cluster_data.copy()
for subject_id in df['subject_id'].unique():
    participant_mask = df['subject_id'] == subject_id
    participant_data = cluster_data[participant_mask]
    if len(participant_data) > 1:  # Only scale if participant has multiple observations
        scaler_participant = StandardScaler()
        cluster_data_scaled[participant_mask] = scaler_participant.fit_transform(participant_data)

print("Cluster Analysis Results:")
print("=" * 60)

# Function to find optimal k using elbow method
def find_optimal_k(data, max_k=10, title="Elbow Method"):
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score needs at least 2 clusters
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
            calinski_scores.append(calinski_harabasz_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(0)
            calinski_scores.append(0)
    
    # Create elbow plot
    fig_elbow = go.Figure()
    
    # Inertia plot
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    # Silhouette score plot (secondary y-axis)
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig_elbow.update_layout(
        title=f'{title} - Optimal K Selection',
        xaxis_title='Number of Clusters (k)',
        yaxis=dict(title='Inertia', side='left'),
        yaxis2=dict(title='Silhouette Score', side='right', overlaying='y'),
        width=800,
        height=600,
        showlegend=True
    )
    fig_elbow.write_html('../../results/Behavior/Clustering/clustering_elbow.html')
    # fig_elbow.write_image('../../results/Behavior/Clustering/clustering_elbow.png')
    fig_elbow.show()
    
    # Find optimal k based on silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_elbow = k_range[np.argmax(np.diff(inertias, 2)) + 1]  # Second derivative method
    
    print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
    print(f"Optimal k (Elbow): {optimal_k_elbow}")
    
    return optimal_k_silhouette, inertias, silhouette_scores

# 1. K-means Clustering
print("\n1. K-MEANS CLUSTERING")
print("-" * 40)

# Find optimal k for K-means
optimal_k_kmeans, inertias_kmeans, sil_scores_kmeans = find_optimal_k(
    cluster_data_scaled, 
    max_k=8, 
    title="K-means Clustering"
)

# Perform K-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k_kmeans, random_state=42, n_init=10)
cluster_labels_kmeans = kmeans.fit_predict(cluster_data_scaled)

# Add cluster labels to data
cluster_data_kmeans = cluster_data.copy()
cluster_data_kmeans['Cluster'] = cluster_labels_kmeans

# Create 3D scatter plot for K-means
fig_3d_kmeans = go.Figure()

for cluster_id in range(optimal_k_kmeans):
    cluster_points = cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id]
    fig_3d_kmeans.add_trace(go.Scatter3d(
        x=cluster_points['valence'],
        y=cluster_points['selfother'],
        z=cluster_points['time'],
        mode='markers',
        name=f'Cluster {cluster_id}',
        marker=dict(size=6, opacity=0.7)
    ))

fig_3d_kmeans.update_layout(
    title=f'K-means Clustering (k={optimal_k_kmeans})',
    scene=dict(
        xaxis_title='Valence',
        yaxis_title='Self/Other',
        zaxis_title='Time'
    ),
    width=800,
    height=600
)
fig_3d_kmeans.write_html('../../results/Behavior/Clustering/clustering_3d_kmeans.html')
# fig_3d_kmeans.write_image('../../results/Behavior/Clustering/clustering_3d_kmeans.png')
fig_3d_kmeans.show()

# 2. Hierarchical Clustering
print("\n2. HIERARCHICAL CLUSTERING")
print("-" * 40)

# Create dendrogram for hierarchical clustering
linkage_matrix = linkage(cluster_data_scaled, method='ward')

# Create dendrogram
fig_dendro = go.Figure()

# Create dendrogram data
def create_dendrogram_data(linkage_matrix, labels=None):
    if labels is None:
        labels = [f'Point {i}' for i in range(len(linkage_matrix) + 1)]
    
    # Extract coordinates for dendrogram
    dendro_data = []
    for i, (idx1, idx2, dist, count) in enumerate(linkage_matrix):
        dendro_data.append({
            'x': [idx1, idx1, idx2, idx2],
            'y': [0, dist, dist, 0],
            'cluster': i
        })
    
    return dendro_data

dendro_data = create_dendrogram_data(linkage_matrix)

for dendro_item in dendro_data:
    fig_dendro.add_trace(go.Scatter(
        x=dendro_item['x'],
        y=dendro_item['y'],
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))

fig_dendro.update_layout(
    title='Hierarchical Clustering Dendrogram',
    xaxis_title='Data Points',
    yaxis_title='Distance',
    width=800,
    height=600
)
fig_dendro.write_html('../../results/Behavior/Clustering/clustering_dendro.html')
# fig_dendro.write_image('../../results/Behavior/Clustering/clustering_dendro.png')
fig_dendro.show()

# Find optimal k for hierarchical clustering using silhouette score
silhouette_scores_hier = []
k_range = range(2, 9)

for k in k_range:
    hierarchical = AgglomerativeClustering(n_clusters=k)
    labels_hier = hierarchical.fit_predict(cluster_data_scaled)
    silhouette_scores_hier.append(silhouette_score(cluster_data_scaled, labels_hier))

optimal_k_hierarchical = k_range[np.argmax(silhouette_scores_hier)]
print(f"Optimal k for Hierarchical Clustering: {optimal_k_hierarchical}")

# Perform hierarchical clustering with optimal k
hierarchical = AgglomerativeClustering(n_clusters=optimal_k_hierarchical)
cluster_labels_hierarchical = hierarchical.fit_predict(cluster_data_scaled)

# Add cluster labels to data
cluster_data_hierarchical = cluster_data.copy()
cluster_data_hierarchical['Cluster'] = cluster_labels_hierarchical

# Create 3D scatter plot for hierarchical clustering
fig_3d_hierarchical = go.Figure()

for cluster_id in range(optimal_k_hierarchical):
    cluster_points = cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id]
    fig_3d_hierarchical.add_trace(go.Scatter3d(
        x=cluster_points['valence'],
        y=cluster_points['selfother'],
        z=cluster_points['time'],
        mode='markers',
        name=f'Cluster {cluster_id}',
        marker=dict(size=6, opacity=0.7)
    ))

fig_3d_hierarchical.update_layout(
    title=f'Hierarchical Clustering (k={optimal_k_hierarchical})',
    scene=dict(
        xaxis_title='Valence',
        yaxis_title='Self/Other',
        zaxis_title='Time'
    ),
    width=800,
    height=600
)
fig_3d_hierarchical.write_html('../../results/Behavior/Clustering/clustering_3d_hierarchical.html')
# fig_3d_hierarchical.write_image('../../results/Behavior/Clustering/clustering_3d_hierarchical.png')
fig_3d_hierarchical.show()

# 3. Comparison Analysis
print("\n3. CLUSTERING COMPARISON")
print("-" * 40)

# Create comparison table
comparison_data = {
    'Method': ['K-means', 'Hierarchical'],
    'Optimal k': [optimal_k_kmeans, optimal_k_hierarchical],
    'Silhouette Score': [max(sil_scores_kmeans), max(silhouette_scores_hier)],
    'N Clusters': [optimal_k_kmeans, optimal_k_hierarchical]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Create cluster size comparison
fig_cluster_sizes = go.Figure()

# K-means cluster sizes
kmeans_cluster_sizes = cluster_data_kmeans['Cluster'].value_counts().sort_index()
fig_cluster_sizes.add_trace(go.Bar(
    x=[f'Cluster {i}' for i in kmeans_cluster_sizes.index],
    y=kmeans_cluster_sizes.values,
    name='K-means',
    marker_color='blue'
))

# Hierarchical cluster sizes
hierarchical_cluster_sizes = cluster_data_hierarchical['Cluster'].value_counts().sort_index()
fig_cluster_sizes.add_trace(go.Bar(
    x=[f'Cluster {i}' for i in hierarchical_cluster_sizes.index],
    y=hierarchical_cluster_sizes.values,
    name='Hierarchical',
    marker_color='red'
))

fig_cluster_sizes.update_layout(
    title='Cluster Size Comparison',
    xaxis_title='Cluster',
    yaxis_title='Number of Observations',
    barmode='group',
    width=800,
    height=500
)
fig_cluster_sizes.write_html('../../results/Behavior/Clustering/clustering_cluster_sizes.html')
# fig_cluster_sizes.write_image('../../results/Behavior/Clustering/clustering_cluster_sizes.png')
fig_cluster_sizes.show()

# 4. Enhanced Cluster Profiling
print("\n" + "=" * 60)
print("DETAILED CLUSTER PROFILING")
print("=" * 60)

# 4.1 K-means Cluster Profiling
print("\n4.1 K-MEANS CLUSTER PROFILES")
print("-" * 50)

# Calculate comprehensive statistics for K-means clusters
cluster_profiles_kmeans = cluster_data_kmeans.groupby('Cluster').agg({
    'valence': ['mean', 'std', 'min', 'max', 'count'],
    'selfother': ['mean', 'std', 'min', 'max', 'count'],
    'time': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

print("Comprehensive Cluster Statistics (K-means):")
print(cluster_profiles_kmeans)

# Create K-means cluster profile visualization
fig_cluster_profiles_kmeans = go.Figure()

# Add radar chart for each K-means cluster
for cluster_id in range(optimal_k_kmeans):
    cluster_means = cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id][cluster_variables].mean()
    
    fig_cluster_profiles_kmeans.add_trace(go.Scatterpolar(
        r=cluster_means.values,
        theta=cluster_variables,
        fill='toself',
        name=f'Cluster {cluster_id}',
        line=dict(width=2)
    ))

fig_cluster_profiles_kmeans.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, cluster_data_kmeans[cluster_variables].max().max()]
        )),
    showlegend=True,
    title=f'K-means Cluster Profiles (k={optimal_k_kmeans})',
    width=800,
    height=600
)
fig_cluster_profiles_kmeans.write_html('../../results/Behavior/Clustering/clustering_cluster_profiles_kmeans.html')
# fig_cluster_profiles_kmeans.write_image('../../results/Behavior/Clustering/clustering_cluster_profiles_kmeans.png')
fig_cluster_profiles_kmeans.show()

# 4.2 Hierarchical Cluster Profiling
print("\n4.2 HIERARCHICAL CLUSTER PROFILES")
print("-" * 50)

# Calculate comprehensive statistics for hierarchical clusters
cluster_profiles_hierarchical = cluster_data_hierarchical.groupby('Cluster').agg({
    'valence': ['mean', 'std', 'min', 'max', 'count'],
    'selfother': ['mean', 'std', 'min', 'max', 'count'],
    'time': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

print("Comprehensive Cluster Statistics (Hierarchical):")
print(cluster_profiles_hierarchical)

# Create hierarchical cluster profile visualization
fig_cluster_profiles_hierarchical = go.Figure()

# Add radar chart for each hierarchical cluster
for cluster_id in range(optimal_k_hierarchical):
    cluster_means = cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id][cluster_variables].mean()
    
    fig_cluster_profiles_hierarchical.add_trace(go.Scatterpolar(
        r=cluster_means.values,
        theta=cluster_variables,
        fill='toself',
        name=f'Cluster {cluster_id}',
        line=dict(width=2)
    ))

fig_cluster_profiles_hierarchical.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, cluster_data_hierarchical[cluster_variables].max().max()]
        )),
    showlegend=True,
    title=f'Hierarchical Cluster Profiles (k={optimal_k_hierarchical})',
    width=800,
    height=600
)
fig_cluster_profiles_hierarchical.write_html('../../results/Behavior/Clustering/clustering_cluster_profiles_hierarchical.html')
#fig_cluster_profiles_hierarchical.write_image('../../results/Behavior/Clustering/clustering_cluster_profiles_hierarchical.png')
fig_cluster_profiles_hierarchical.show()

# 5. Cluster Interpretation
print("\n5. CLUSTER INTERPRETATION")
print("-" * 50)

# Function to interpret clusters based on their characteristics
def interpret_cluster(cluster_means, data_reference):
    valence_mean = cluster_means['valence']
    selfother_mean = cluster_means['selfother']
    time_mean = cluster_means['time']
    
    # Valence interpretation
    if valence_mean > data_reference['valence'].mean():
        valence_desc = "High valence"
    else:
        valence_desc = "Low valence"
        
    # Self/Other interpretation
    if selfother_mean > data_reference['selfother'].mean():
        selfother_desc = "More self-focused"
    else:
        selfother_desc = "More other-focused"
        
    # Time interpretation
    if time_mean > data_reference['time'].mean():
        time_desc = "Future-oriented"
    else:
        time_desc = "Past/present-oriented"
        
    return f"{valence_desc}, {selfother_desc}, {time_desc}"

# Interpret K-means clusters
print("K-means Cluster Interpretations:")
for cluster_id in range(optimal_k_kmeans):
    cluster_means = cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id][cluster_variables].mean()
    cluster_size = len(cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id])
    interpretation = interpret_cluster(cluster_means, cluster_data)
    
    print(f"Cluster {cluster_id} (n={cluster_size}): {interpretation}")
    print(f"  Valence: {cluster_means['valence']:.3f}")
    print(f"  Self/Other: {cluster_means['selfother']:.3f}")
    print(f"  Time: {cluster_means['time']:.3f}")
    print()

# Interpret hierarchical clusters
print("Hierarchical Cluster Interpretations:")
for cluster_id in range(optimal_k_hierarchical):
    cluster_means = cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id][cluster_variables].mean()
    cluster_size = len(cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id])
    interpretation = interpret_cluster(cluster_means, cluster_data)
    
    print(f"Cluster {cluster_id} (n={cluster_size}): {interpretation}")
    print(f"  Valence: {cluster_means['valence']:.3f}")
    print(f"  Self/Other: {cluster_means['selfother']:.3f}")
    print(f"  Time: {cluster_means['time']:.3f}")
    print()

# 6. Cluster Separation Analysis
print("\n6. CLUSTER SEPARATION ANALYSIS")
print("-" * 50)

# For K-means
cluster_centers_kmeans = kmeans.cluster_centers_
distances_kmeans = cdist(cluster_centers_kmeans, cluster_centers_kmeans)
np.fill_diagonal(distances_kmeans, np.inf)
min_distances_kmeans = np.min(distances_kmeans, axis=1)

print("Cluster Separation (K-means):")
for i in range(optimal_k_kmeans):
    print(f"Cluster {i}: Minimum distance to other clusters = {min_distances_kmeans[i]:.3f}")

# For hierarchical clustering (calculate centers manually)
hierarchical_centers = []
for cluster_id in range(optimal_k_hierarchical):
    cluster_points = cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id][cluster_variables]
    center = cluster_points.mean().values
    hierarchical_centers.append(center)

hierarchical_centers = np.array(hierarchical_centers)
distances_hierarchical = cdist(hierarchical_centers, hierarchical_centers)
np.fill_diagonal(distances_hierarchical, np.inf)
min_distances_hierarchical = np.min(distances_hierarchical, axis=1)

print("\nCluster Separation (Hierarchical):")
for i in range(optimal_k_hierarchical):
    print(f"Cluster {i}: Minimum distance to other clusters = {min_distances_hierarchical[i]:.3f}")

# 7. Save Results
print("\n7. SAVING RESULTS")
print("-" * 50)

# Save clustering results
# Merge with original data to include subject, task, and probe_id information
cluster_data_kmeans_with_metadata = pd.merge(
    cluster_data_kmeans, 
    df[['subject_id', 'task', 'probe_number'] + cluster_variables], 
    on=cluster_variables, 
    how='left'
)
cluster_data_hierarchical_with_metadata = pd.merge(
    cluster_data_hierarchical, 
    df[['subject_id', 'task', 'probe_number'] + cluster_variables], 
    on=cluster_variables, 
    how='left'
)

cluster_data_kmeans_with_metadata.to_csv('../../results/Behavior/Clustering/clustering_kmeans.csv', index=False)
cluster_data_hierarchical_with_metadata.to_csv('../../results/Behavior/Clustering/clustering_hierarchical.csv', index=False)
comparison_df.to_csv('../../results/Behavior/Clustering/clustering_comparison.csv', index=False)

# Save detailed cluster profiles
cluster_profiles_kmeans.to_csv('../../results/Behavior/Clustering/cluster_profiles_kmeans.csv')
cluster_profiles_hierarchical.to_csv('../../results/Behavior/Clustering/cluster_profiles_hierarchical.csv')

# Create summary table with interpretations
summary_data = []
for cluster_id in range(optimal_k_kmeans):
    cluster_means = cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id][cluster_variables].mean()
    cluster_size = len(cluster_data_kmeans[cluster_data_kmeans['Cluster'] == cluster_id])
    interpretation = interpret_cluster(cluster_means, cluster_data)
    
    summary_data.append({
        'Method': 'K-means',
        'Cluster': cluster_id,
        'Size': cluster_size,
        'Valence_Mean': cluster_means['valence'],
        'SelfOther_Mean': cluster_means['selfother'],
        'Time_Mean': cluster_means['time'],
        'Interpretation': interpretation
    })

for cluster_id in range(optimal_k_hierarchical):
    cluster_means = cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id][cluster_variables].mean()
    cluster_size = len(cluster_data_hierarchical[cluster_data_hierarchical['Cluster'] == cluster_id])
    interpretation = interpret_cluster(cluster_means, cluster_data)
    
    summary_data.append({
        'Method': 'Hierarchical',
        'Cluster': cluster_id,
        'Size': cluster_size,
        'Valence_Mean': cluster_means['valence'],
        'SelfOther_Mean': cluster_means['selfother'],
        'Time_Mean': cluster_means['time'],
        'Interpretation': interpretation
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../../results/Behavior/Clustering/cluster_interpretations_summary.csv', index=False)

print("Clustering results saved to:")
print("- clustering_kmeans.csv")
print("- clustering_hierarchical.csv")
print("- clustering_comparison.csv")
print("- cluster_profiles_kmeans.csv")
print("- cluster_profiles_hierarchical.csv")
print("- cluster_interpretations_summary.csv")

# Summary
print("\n" + "=" * 60)
print("CLUSTERING ANALYSIS SUMMARY")
print("=" * 60)
print(f"K-means: {optimal_k_kmeans} clusters, Silhouette Score: {max(sil_scores_kmeans):.3f}")
print(f"Hierarchical: {optimal_k_hierarchical} clusters, Silhouette Score: {max(silhouette_scores_hier):.3f}")
print(f"Best method: {'Hierarchical' if max(silhouette_scores_hier) > max(sil_scores_kmeans) else 'K-means'}")
print("=" * 60)

# %%
