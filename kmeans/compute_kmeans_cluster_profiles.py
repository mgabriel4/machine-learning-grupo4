import pandas as pd, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

base_dir = os.path.dirname(__file__)
X_path = os.path.join(base_dir,'kmeans_X.csv')
clusters_path = os.path.join(base_dir,'kmeans_clusters.csv')
model_path = os.path.join(base_dir,'kmeans_model.joblib')

X = pd.read_csv(X_path)
X_cols = X.columns.tolist()

# Reajustar scaler para perfis em z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reproduzir clusters se não existir arquivo de labels (ou ler)
if os.path.exists(clusters_path):
    clusters_df = pd.read_csv(clusters_path)
    if 'cluster' in clusters_df.columns:
        labels = clusters_df['cluster'].values
    else:
        labels = clusters_df.iloc[:,0].values
else:
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

import numpy as np
labels_series = pd.Series(labels, name='cluster')
cluster_sizes = labels_series.value_counts().sort_index()
size_pct = (cluster_sizes/len(X)*100).round(2)

# Centróides em espaço original e z-score médio dos clusters
centroids_z = []
for c in sorted(cluster_sizes.index):
    mask = labels==c
    centroid = X_scaled[mask].mean(axis=0)
    centroids_z.append(centroid)
centroids_z = np.vstack(centroids_z)

# Diferença absoluta entre centróides em z-score
if centroids_z.shape[0] == 2:
    diff_abs = np.abs(centroids_z[0]-centroids_z[1])
    top_diff_idx = diff_abs.argsort()[::-1][:10]
    top_diff = [(X_cols[i], float(diff_abs[i])) for i in top_diff_idx]
else:
    top_diff = []

result = {
  'cluster_sizes': {int(k): int(v) for k,v in cluster_sizes.items()},
  'cluster_sizes_pct': {int(k): float(size_pct[k]) for k in cluster_sizes.index},
  'top_feature_differences_z': top_diff
}
print(json.dumps(result, indent=2))
