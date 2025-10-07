import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, silhouette_score
from textwrap import dedent

root = os.getcwd()
# Paths
knn_root = os.path.join(root, 'docs', 'knn')
arvore_root = os.path.join(root, 'docs', 'arvore_decisao')
kmeans_root = os.path.join(root, 'docs', 'kmeans')
rel_path = os.path.join(root, 'docs', 'relatorio_final')
os.makedirs(rel_path, exist_ok=True)

# --- KNN metrics ---
knn_y_test_path = os.path.join(knn_root, 'knn_y_test.csv')
knn_y_pred_path = os.path.join(knn_root, 'knn_y_pred.csv')
knn_metrics = {}
if os.path.exists(knn_y_test_path) and os.path.exists(knn_y_pred_path):
    y_test = pd.read_csv(knn_y_test_path).iloc[:,0]
    y_pred = pd.read_csv(knn_y_pred_path).iloc[:,0]
    knn_metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    knn_metrics['f1'] = float(f1_score(y_test, y_pred))
    knn_metrics['precision'] = float(precision_score(y_test, y_pred))
    knn_metrics['recall'] = float(recall_score(y_test, y_pred))
    knn_metrics['report'] = classification_report(y_test, y_pred)
else:
    knn_metrics['error'] = 'KNN y_test or y_pred CSV missing'

# --- Decision Tree summary ---
feat_imp_path = os.path.join(arvore_root, 'feature_importances_arvore.csv')
grid_path = os.path.join(arvore_root, 'grid_cv_results_arvore.csv')
arvore_summary = {}
if os.path.exists(feat_imp_path):
    fi = pd.read_csv(feat_imp_path, index_col=0)
    # top 10
    fi_sorted = fi.sort_values('importance', ascending=False).head(10)
    arvore_summary['top_features'] = fi_sorted.to_dict()['importance']
else:
    arvore_summary['top_features'] = None

if os.path.exists(grid_path):
    try:
        grid = pd.read_csv(grid_path)
        # take best mean_test_score
        if 'mean_test_score' in grid.columns:
            best_row = grid.loc[grid['mean_test_score'].idxmax()]
            arvore_summary['best_mean_test_score'] = float(best_row['mean_test_score'])
            arvore_summary['best_params'] = {}
            for c in grid.columns:
                if c.startswith('param_'):
                    arvore_summary['best_params'][c.replace('param_','')] = best_row[c]
    except Exception as e:
        arvore_summary['grid_error'] = str(e)
else:
    arvore_summary['grid'] = None

# false positives count if file exists
false_pos_path = os.path.join(arvore_root, 'false_positives_arvore.csv')
if os.path.exists(false_pos_path):
    fp = pd.read_csv(false_pos_path)
    arvore_summary['false_positives_count'] = len(fp)
else:
    arvore_summary['false_positives_count'] = None

# --- KMeans summary ---
kmeans_summary = {}
kmeans_clusters_path = os.path.join(kmeans_root, 'kmeans_clusters.csv')
kmeans_X_path = os.path.join(kmeans_root, 'kmeans_X.csv')
if os.path.exists(kmeans_clusters_path) and os.path.exists(kmeans_X_path):
    X = pd.read_csv(kmeans_X_path)
    labels = pd.read_csv(kmeans_clusters_path).iloc[:,0]
    from sklearn.metrics import silhouette_score
    try:
        kmeans_summary['silhouette'] = float(silhouette_score(X, labels))
    except Exception as e:
        kmeans_summary['silhouette_error'] = str(e)
    # inertia: compute within-cluster sum of squares
    try:
        centers = X.groupby(labels).mean()
        inertia = 0.0
        for lab in labels.unique():
            pts = X[labels==lab].to_numpy()
            center = centers.loc[lab].to_numpy()
            inertia += ((pts - center)**2).sum()
        kmeans_summary['inertia_estimate'] = float(inertia)
    except Exception as e:
        kmeans_summary['inertia_error'] = str(e)
else:
    kmeans_summary['error'] = 'kmeans artifacts missing'

# Build markdown
md = []
md.append('# Relatório Final - Projeto Machine Learning')
md.append('\n## Resumo Executivo')
md.append('\nEste relatório consolida os resultados dos três fluxos implementados: Árvore de Decisão, K-Nearest Neighbors (KNN) e K-Means.')

# KNN section
md.append('\n---\n\n## K-Nearest Neighbors (KNN) - Resultados')
if 'error' in knn_metrics:
    md.append('\nErro ao carregar resultados do KNN: ' + knn_metrics['error'])
else:
    md.append(f"\n- Accuracy: **{knn_metrics['accuracy']:.4f}**")
    md.append(f"\n- Precision: **{knn_metrics['precision']:.4f}**")
    md.append(f"\n- Recall: **{knn_metrics['recall']:.4f}**")
    md.append(f"\n- F1-score: **{knn_metrics['f1']:.4f}**")
    md.append('\n\n### Classification Report\n')
    md.append('```\n' + knn_metrics['report'] + '\n```')
    md.append('\nImagens relevantes:')
    md.append('\n- ![Confusion Matrix](../knn/imagens/knn_confusion_matrix.png)')
    md.append('\n- ![ROC Curve](../knn/imagens/knn_roc.png)')
    md.append('\n- ![Precision-Recall Curve](../knn/imagens/knn_pr_curve.png)')

# Decision Tree section
md.append('\n---\n\n## Árvore de Decisão - Resultados')
if arvore_summary.get('top_features') is None:
    md.append('\nSem informações de feature importance disponíveis.')
else:
    md.append('\n### Top features (importância)')
    md.append('\n| Feature | Importance |')
    md.append('\n|---|---:|')
    for f, imp in arvore_summary['top_features'].items():
        md.append(f"\n| {f} | {imp:.5f} |")

if 'best_mean_test_score' in arvore_summary:
    md.append(f"\n- Melhor mean_test_score (GridSearch): **{arvore_summary['best_mean_test_score']:.4f}**")
    if arvore_summary.get('best_params'):
        md.append('\n- Parâmetros do melhor modelo:')
        for k,v in arvore_summary['best_params'].items():
            md.append(f"  - {k}: {v}")

md.append(f"\n- False positives registrados (arquivo): {arvore_summary.get('false_positives_count')}")
md.append('\nImagens relevantes:')
md.append('\n- ![Árvore reduzida](../arvore_decisao/imagens/arvore_reduzida.png)')
md.append('\n- ![Feature importances](../arvore_decisao/imagens/feature_importances.png)')

# KMeans section
md.append('\n---\n\n## K-Means - Resultados')
if 'error' in kmeans_summary:
    md.append('\nErro ao carregar resultados do KMeans: ' + kmeans_summary['error'])
else:
    if 'silhouette' in kmeans_summary:
        md.append(f"\n- Silhouette score (clusters): **{kmeans_summary['silhouette']:.4f}**")
    if 'inertia_estimate' in kmeans_summary:
        md.append(f"\n- Inertia (est.): **{kmeans_summary['inertia_estimate']:.2f}**")
    md.append('\nImagens relevantes:')
    md.append('\n- ![Elbow](../kmeans/imagens/kmeans_elbow.png)')
    md.append('\n- ![Silhouette](../kmeans/imagens/kmeans_silhouette.png)')
    md.append('\n- ![Clusters PCA](../kmeans/imagens/kmeans_clusters.png)')

# Conclusions and recommendations
md.append('\n---\n\n## Conclusões e Recomendações')
md.append('\n- O fluxo está implementado e reproduzível dentro do repositório; artefatos (modelos, imagens, CSVs) foram salvos em `docs/*`.')
md.append('\n- Observação: o KNN obteve F1 baixo; recomenda-se testar balanceamento e algoritmos alternativos (RandomForest, XGBoost) e engenharia de features.')
md.append('\n- Para a Árvore de Decisão, analisar false positives e considerar tuning adicional ou poda mais rígida.')
md.append('\n- Para o KMeans, explorar incluir variáveis categóricas codificadas e tentar clusters adicionais.')

# Write file
out_file = os.path.join(rel_path, 'relatorio_final.md')
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print('Relatório gerado em', out_file)
