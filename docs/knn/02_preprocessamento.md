# 02 - Pré-processamento

Pré-processamento aplicado antes do treinamento do KNN:

- Seleção de colunas numéricas e categóricas
- Imputação (mediana para numéricas, moda para categóricas)
- One-Hot Encoding para categóricas
- Escalonamento (StandardScaler) para numéricas

O pipeline está implementado no notebook `knn.ipynb` e usa `ColumnTransformer`.
