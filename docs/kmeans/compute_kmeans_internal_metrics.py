import pandas as pd, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

base_dir = os.path.dirname(__file__)
X_path = os.path.join(base_dir,'kmeans_X.csv')
X = pd.read_csv(X_path)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []
prev_inertia = None
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    rel_drop = None
    if prev_inertia is not None:
        rel_drop = (prev_inertia - inertia)/prev_inertia
    results.append({
        'k': k,
        'inertia': inertia,
        'silhouette': sil,
        'davies_bouldin': db,
        'rel_inertia_drop_from_prev': rel_drop
    })
    prev_inertia = inertia
print(json.dumps(results, indent=2))
