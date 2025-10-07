import pandas as pd, json, numpy as np
from pathlib import Path

# Localização preferencial dentro da pasta knn
paths = [
    Path('docs/knn/funcionarios.csv'),
    Path('docs/arvore_decisao/funcionarios.csv')
]
for p in paths:
    if p.exists():
        data_path = p
        break
else:
    raise FileNotFoundError('Arquivo funcionarios.csv não encontrado em caminhos padrão.')

df = pd.read_csv(data_path)
rows, cols = df.shape

# Distribuição do alvo
if 'Attrition' in df.columns:
    target_counts = df['Attrition'].value_counts().to_dict()
    target_pct = (df['Attrition'].value_counts(normalize=True)*100).round(2).to_dict()
else:
    target_counts, target_pct = {}, {}

# Tipos
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

# Cardinalidade (apenas categóricas)
cardinality = {c: df[c].nunique() for c in cat_cols}
cardinality_top = sorted(cardinality.items(), key=lambda x: x[1], reverse=True)[:10]

# Missing
missing = df.isna().sum()
missing_pct = (missing / rows * 100).round(2)
missing_rows = [
    {
        'col': c,
        'missing': int(missing[c]),
        'pct': float(missing_pct[c])
    }
    for c in df.columns if missing[c] > 0
]

# Estatísticas numéricas resumidas
num_summary = df[numeric_cols].describe().T[['mean','std','min','max']].round(2)
key_vars = [c for c in ['Age','MonthlyIncome','DistanceFromHome','YearsAtCompany','YearsInCurrentRole'] if c in df.columns]
key_stats = num_summary.loc[key_vars].to_dict('index') if key_vars else {}

out = {
    'shape': {'rows': rows, 'cols': cols},
    'n_numeric': len(numeric_cols),
    'n_categorical': len(cat_cols),
    'target_counts': target_counts,
    'target_pct': target_pct,
    'missing_columns': missing_rows,
    'cardinality_top': cardinality_top,
    'key_numeric_stats': key_stats,
    'data_path': str(data_path)
}
print(json.dumps(out, indent=2, ensure_ascii=False))
