"""Script unificado para reproduzir pipeline completo.
Uso:
    python run_all.py --skip-kmeans  # exemplo para pular clustering
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from ml.data_loading import load_employees
from ml.preprocessing import split_feature_types, build_preprocessor
from ml.models import build_knn_pipeline, build_decision_tree_pipeline, default_knn_param_grid, default_tree_param_grid, train_kmeans
from ml.evaluation import evaluate_classifier, evaluate_clusters
from ml.reporting import generate_classification_markdown, generate_cluster_markdown, write_auto_metrics

ROOT = Path(__file__).parent
DOCS = ROOT / 'docs'
KNN_DIR = DOCS / 'knn'
TREE_DIR = DOCS / 'arvore_decisao'
KMEANS_DIR = DOCS / 'kmeans'
REL_DIR = DOCS / 'relatorio_final'

TARGET = 'Attrition'


def train_classifier(model_name: str, estimator, param_grid, X, y):
    pipe = build_knn_pipeline(preprocessor) if model_name=='KNN' else build_decision_tree_pipeline(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X, y)
    return gs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-kmeans', action='store_true')
    parser.add_argument('--kmeans-k', type=int, default=2)
    args = parser.parse_args()

    csv_path = KNN_DIR / 'funcionarios.csv'
    df = load_employees(csv_path)
    # Encode target
    y = df[TARGET].map({'Yes':1, 'No':0})
    num_cols, cat_cols = split_feature_types(df, TARGET)
    X = df[num_cols + cat_cols].copy()

    global preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # KNN
    knn_grid = default_knn_param_grid()
    knn_pipe = build_knn_pipeline(preprocessor)
    knn_cv = GridSearchCV(knn_pipe, knn_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1', n_jobs=-1)
    knn_cv.fit(X_train, y_train)
    knn_best = knn_cv.best_estimator_
    KNN_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(knn_best, KNN_DIR / 'knn_model.joblib')
    y_pred_knn = knn_best.predict(X_test)
    pd.DataFrame({'Attrition': y_test.reset_index(drop=True)}).to_csv(KNN_DIR / 'knn_y_test.csv', index=False)
    pd.DataFrame({'y_pred': y_pred_knn}).to_csv(KNN_DIR / 'knn_y_pred.csv', index=False)
    knn_metrics = evaluate_classifier(y_test.values, y_pred_knn)

    # Decision Tree
    tree_grid = default_tree_param_grid()
    tree_pipe = build_decision_tree_pipeline(preprocessor)
    tree_cv = GridSearchCV(tree_pipe, tree_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1', n_jobs=-1)
    tree_cv.fit(X_train, y_train)
    tree_best = tree_cv.best_estimator_
    TREE_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(tree_best, TREE_DIR / 'arvore_decisao_pipeline.joblib')
    y_pred_tree = tree_best.predict(X_test)
    tree_metrics = evaluate_classifier(y_test.values, y_pred_tree)

    # KMeans
    cluster_md = ''
    if not args.skip_kmeans:
        # Para clustering usamos apenas numéricos escalados do preprocessor
        # Ajustar preprocessor nos dados completos para obter matriz transformada
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        # X_transformed é numpy; salvar subset para rastreabilidade
        kmeans_input = pd.DataFrame(X_transformed)
        KMEANS_DIR.mkdir(exist_ok=True, parents=True)
        kmeans_input.to_csv(KMEANS_DIR / 'kmeans_X.csv', index=False)
        km_model, labels = train_kmeans(kmeans_input, n_clusters=args.kmeans_k)
        joblib.dump(km_model, KMEANS_DIR / 'kmeans_model.joblib')
        pd.DataFrame({'cluster': labels}).to_csv(KMEANS_DIR / 'kmeans_clusters.csv', index=False)
        cluster_metrics = evaluate_clusters(kmeans_input, labels)
        cluster_md = generate_cluster_markdown('K-Means', cluster_metrics)
        # Persist scaler do bloco numérico para reutilização (se existir)
        # Acessar pipeline internamente
        try:
            num_scaler = preprocessor.named_transformers_['num'].named_steps.get('scaler')
            if num_scaler:
                joblib.dump(num_scaler, KMEANS_DIR / 'kmeans_scaler.joblib')
        except Exception:
            pass
    else:
        cluster_md = 'K-Means pulado.'

    # Gerar markdown automático
    REL_DIR.mkdir(exist_ok=True, parents=True)
    sections = [
        generate_classification_markdown('KNN', knn_metrics),
        generate_classification_markdown('Árvore de Decisão', tree_metrics),
        cluster_md
    ]
    write_auto_metrics(REL_DIR / 'auto_metrics.md', sections)
    print('Concluído. Métricas escritas em relatorio_final/auto_metrics.md')
