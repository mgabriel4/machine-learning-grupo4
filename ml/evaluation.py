from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)


def evaluate_classifier(y_true, y_pred) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return {
        'accuracy': acc,
        'per_class': {
            str(i): {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i in range(len(precision))
        },
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'confusion_matrix': cm.tolist(),
        'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
    }


def evaluate_clusters(X, labels) -> Dict[str, Any]:
    out = {}
    try:
        out['silhouette'] = float(silhouette_score(X, labels))
    except Exception as e:
        out['silhouette_error'] = str(e)
    try:
        out['davies_bouldin'] = float(davies_bouldin_score(X, labels))
    except Exception as e:
        out['davies_bouldin_error'] = str(e)
    try:
        out['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
    except Exception as e:
        out['calinski_harabasz_error'] = str(e)
    return out
