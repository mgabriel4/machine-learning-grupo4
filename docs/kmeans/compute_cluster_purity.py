import pandas as pd, json, os
base_dir = os.path.dirname(__file__)
orig_path = os.path.join(base_dir,'funcionarios.csv')
clusters_path = os.path.join(base_dir,'kmeans_clusters.csv')

orig = pd.read_csv(orig_path)
cl = pd.read_csv(clusters_path)
if 'cluster' not in cl.columns:
    cl.columns = ['cluster']

df = pd.concat([orig, cl], axis=1)
res = {}
for c, sub in df.groupby('cluster'):
    vc = sub['Attrition'].value_counts()
    pct = (vc/len(sub)*100).round(2)
    res[int(c)] = {
        'size': int(len(sub)),
        'attrition_counts': vc.to_dict(),
        'attrition_pct': pct.to_dict()
    }
print(json.dumps(res, indent=2, ensure_ascii=False))
