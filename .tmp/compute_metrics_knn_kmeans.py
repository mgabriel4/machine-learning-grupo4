import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

out = {}
# KNN
knn_test = 'docs/knn/knn_y_test.csv'
knn_pred = 'docs/knn/knn_y_pred.csv'
if os.path.exists(knn_test) and os.path.exists(knn_pred):
    y_test = pd.read_csv(knn_test).iloc[:,0]
    y_pred = pd.read_csv(knn_pred).iloc[:,0]
    out['knn'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'support': int(len(y_test)),
        'report': classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }
else:
    out['knn'] = None

# KMeans: try to compute silhouette if possible (we may not have labels for true clusters)
kmeans_X = 'docs/kmeans/kmeans_X.csv'
kmeans_clusters = 'docs/kmeans/kmeans_clusters.csv'
if os.path.exists(kmeans_X) and os.path.exists(kmeans_clusters):
    try:
        from sklearn.metrics import silhouette_score
        X = pd.read_csv(kmeans_X)
        labels = pd.read_csv(kmeans_clusters).iloc[:,0]
        sil = float(silhouette_score(X, labels))
        out['kmeans'] = {'silhouette': sil}
    except Exception as e:
        out['kmeans'] = {'error': str(e)}
else:
    out['kmeans'] = None

# Decision tree: we may not have saved test preds; try to find a file
# Search for *_y_test or arvore preds
# For now leave decision tree as None
out['decision_tree'] = None

print(json.dumps(out))
