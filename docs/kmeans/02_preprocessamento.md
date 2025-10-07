# 02 - Pré-processamento (K-Means)

Objetivo

- Preparar as features numéricas para K-Means: seleção, imputação e escalonamento.

Arquivo processado

- `kmeans_X.csv` — conjunto de features numéricas processadas (imputadas e sem identificadores).

Trechos de código

```python
# Selecionar numéricas e remover identificadores
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in ['EmployeeNumber','EmployeeCount','StandardHours']:
    if c in num_cols: num_cols.remove(c)
X = df[num_cols].copy()
X = X.fillna(X.median())

# Escalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Salvar X processado
pd.DataFrame(X, columns=num_cols).to_csv('kmeans_X.csv', index=False)
```

Observações

- Atualmente o pipeline utiliza apenas variáveis numéricas; incluir categóricas (one-hot) pode melhorar resultados.
