import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

yt = pd.read_csv('docs/knn/knn_y_test.csv')['Attrition']
yp = pd.read_csv('docs/knn/knn_y_pred.csv')['y_pred']

metrics = {
    'n_test': len(yt),
    'accuracy': float(accuracy_score(yt, yp)),
    'f1': float(f1_score(yt, yp)),
    'precision': float(precision_score(yt, yp)),
    'recall': float(recall_score(yt, yp)),
}

report = classification_report(yt, yp)
print(json.dumps({'metrics': metrics, 'report': report}, ensure_ascii=False))
