import pandas as pd, json, joblib
from pathlib import Path

csv_path = Path('docs/arvore_decisao/grid_cv_results_arvore.csv')
model_path = Path('docs/arvore_decisao/arvore_decisao_pipeline.joblib')

df = pd.read_csv(csv_path)
# Ordenar por rank (já está) só por garantia
if 'rank_test_score' in df.columns:
    df = df.sort_values('rank_test_score')

# Top 5
top5 = df.nsmallest(5, 'rank_test_score').copy()
# Calcular gap
top5['gap_train_test'] = top5['mean_train_score'] - top5['mean_test_score']

# Estatísticas globais
stats = df['mean_test_score'].describe().to_dict()

# Gaps globais
gap_series = df['mean_train_score'] - df['mean_test_score']
Gap = {
    'mean_gap': float(gap_series.mean()),
    'median_gap': float(gap_series.median()),
    'min_gap': float(gap_series.min()),
    'max_gap': float(gap_series.max())
}

# Estabilidade entre top 10
top10 = df.nsmallest(10,'rank_test_score').copy()
most_stable = top10.loc[top10['std_test_score'].idxmin()].copy()
most_stable_gap = float(most_stable['mean_train_score'] - most_stable['mean_test_score'])

# Complexidade do modelo salvo
model_info = {}
if model_path.exists():
    pipeline = joblib.load(model_path)
    clf = getattr(pipeline, 'named_steps', {}).get('clf')
    if clf is not None and hasattr(clf, 'tree_'):
        import numpy as np
        tree = clf.tree_
        leaves = int((tree.children_left == -1).sum())
        model_info = {
            'criterion': getattr(clf, 'criterion', None),
            'max_depth_param': getattr(clf, 'max_depth', None),
            'min_samples_split_param': getattr(clf, 'min_samples_split', None),
            'depth_actual': int(clf.get_depth()),
            'node_count': int(tree.node_count),
            'leaves': leaves,
            'leaf_ratio': round(leaves / tree.node_count, 4)
        }

result = {
    'n_combinations': int(len(df)),
    'cv_folds': 5,
    'total_fits': int(len(df) * 5),
    'top5': top5[['rank_test_score','param_clf__criterion','param_clf__max_depth','param_clf__min_samples_split','mean_test_score','std_test_score','mean_train_score','gap_train_test']].to_dict(orient='records'),
    'global_stats_mean_test_score': {k: float(v) for k,v in stats.items()},
    'gap_overall': Gap,
    'most_stable_top10': {
        'rank': int(most_stable['rank_test_score']),
        'criterion': most_stable['param_clf__criterion'],
        'max_depth': None if pd.isna(most_stable['param_clf__max_depth']) else most_stable['param_clf__max_depth'],
        'min_samples_split': int(most_stable['param_clf__min_samples_split']),
        'mean_test_score': float(most_stable['mean_test_score']),
        'std_test_score': float(most_stable['std_test_score']),
        'mean_train_score': float(most_stable['mean_train_score']),
        'gap_train_test': most_stable_gap
    },
    'model_complexity': model_info
}
import sys, traceback
try:
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
