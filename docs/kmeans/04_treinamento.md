# 04 - Treinamento final (K-Means)

Objetivo

- Treinar o modelo K-Means final (k=2 por padrão) e salvar artefatos para análise posterior.

Arquivos gerados

- `kmeans_clusters.csv` — labels dos clusters.
- `kmeans_X.csv` — features numéricas processadas.
- `kmeans_model.joblib` — modelo serializado.

Trechos de código

```python
from sklearn.cluster import KMeans
import joblib

K_FINAL = 2
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
labels_final = km_final.fit_predict(X_scaled)

# Salvar
pd.DataFrame(labels_final, columns=['cluster']).to_csv('kmeans_clusters.csv', index=False)
pd.DataFrame(X, columns=num_cols).to_csv('kmeans_X.csv', index=False)
joblib.dump(km_final, 'kmeans_model.joblib')
```

Observações

- Os artefatos permitem unir os clusters ao dataset original e explorar perfis (ex.: `pd.concat([df, clusters], axis=1)`).
