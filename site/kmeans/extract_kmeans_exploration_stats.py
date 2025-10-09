import pandas as pd, json, os

base_dir = os.path.dirname(__file__)
func_path = os.path.join(base_dir,'funcionarios.csv')
clusters_path = os.path.join(base_dir,'kmeans_clusters.csv')
X_path = os.path.join(base_dir,'kmeans_X.csv')

summary = {}
df = pd.read_csv(func_path)
summary['shape'] = {'rows': df.shape[0], 'cols': df.shape[1]}
summary['dtypes_count'] = {str(k): int(v) for k,v in df.dtypes.value_counts().items()}
missing_total = int(df.isna().sum().sum())
summary['missing_total'] = missing_total
miss = df.isna().sum()
summary['missing_by_col_nonzero'] = miss[miss>0].to_dict()
cat_cols = [c for c in df.columns if df[c].dtype == object]
card = {c: int(df[c].nunique()) for c in cat_cols}
summary['categorical_cardinality'] = card
# target
if 'Attrition' in df.columns:
    vc = df['Attrition'].value_counts()
    summary['target_counts'] = vc.to_dict()
    summary['target_pct'] = {k: round(v/len(df)*100,2) for k,v in vc.items()}
# numeric stats sample
num_cols = df.select_dtypes(include='number').columns
num_stats = df[num_cols].agg(['mean','std','min','max']).T.sort_index().head(10)
summary['numeric_sample_stats'] = num_stats.round(2).to_dict('index')
# clusters
if os.path.exists(clusters_path):
    cl = pd.read_csv(clusters_path)
    cl_col = None
    for cand in ['cluster','Cluster','cluster_id','kmeans_cluster']:
        if cand in cl.columns:
            cl_col = cand; break
    if cl_col:
        cvc = cl[cl_col].value_counts().sort_index()
        summary['cluster_distribution'] = {str(k): int(v) for k,v in cvc.items()}
        summary['n_clusters_detected'] = int(cvc.index.nunique())
# X transformed
if os.path.exists(X_path):
    X = pd.read_csv(X_path)
    summary['post_processing_shape'] = {'rows': X.shape[0], 'cols': X.shape[1]}
print(json.dumps(summary, indent=2, ensure_ascii=False))
