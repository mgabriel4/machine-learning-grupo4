import pandas as pd, json, joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np

y_true_path = Path('docs/knn/knn_y_test.csv')
y_pred_path = Path('docs/knn/knn_y_pred.csv')
model_path = Path('docs/knn/knn_model.joblib')

y_true = pd.read_csv(y_true_path)['Attrition'].values
y_pred = pd.read_csv(y_pred_path)['y_pred'].values

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
# cm layout sklearn: [[TN FP],[FN TP]]
TN, FP, FN, TP = cm.ravel()

unique_t, counts_t = np.unique(y_true, return_counts=True)
unique_p, counts_p = np.unique(y_pred, return_counts=True)
class_dist_test = dict(zip(map(str, unique_t), map(int, counts_t)))
class_dist_pred = dict(zip(map(str, unique_p), map(int, counts_p)))

model_info = {}
if model_path.exists():
    model = joblib.load(model_path)
    from sklearn.neighbors import KNeighborsClassifier
    if hasattr(model, 'named_steps'):
        knn = None
        for v in model.named_steps.values():
            if isinstance(v, KNeighborsClassifier):
                knn = v
                break
    else:
        knn = model if isinstance(model, KNeighborsClassifier) else None
    if knn:
        model_info = {
            'n_neighbors': knn.n_neighbors,
            'weights': knn.weights,
            'metric': knn.metric,
            'p': getattr(knn, 'p', None),
            'algorithm': knn.algorithm,
            'leaf_size': knn.leaf_size,
            'effective_metric_params': getattr(knn, 'effective_metric_params_', None)
        }

out = {
  'n_test': int(len(y_true)),
  'class_distribution_test': class_dist_test,
  'class_distribution_pred': class_dist_pred,
  'accuracy': acc,
  'per_class': {
      '0': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': int(support[0])},
      '1': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': int(support[1])}
  },
  'macro_avg_f1': report['macro avg']['f1-score'],
  'weighted_avg_f1': report['weighted avg']['f1-score'],
  'confusion_matrix': cm.tolist(),
  'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP),
  'model_info': model_info
}
print(json.dumps(out, indent=2))
