# 02 - Pré-processamento

Objetivo

- Preparar os dados para o modelo: imputação, encoding de categóricas, remoção de colunas irrelevantes e escalonamento das numéricas.

Arquivo processado

- `arvore_decisao_processed.csv` — dataset resultante após aplicar imputação e encoding (salvo pelo notebook).

Trechos de código relevantes

```python
# Mapear alvo e remover colunas
df['Attrition_binary'] = df['Attrition'].map({'Yes':1,'No':0})
drop_cols = ['EmployeeCount','EmployeeNumber','Over18','StandardHours','Attrition']
df_proc = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Identificar colunas
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Aplicar e salvar
X_proc = preprocessor.fit_transform(X)
# Salvar como CSV (transformado para DataFrame com nomes de features)
```

Resultados

- Número de features após encoding: ver `arvore_decisao_processed.csv` (por ex. 51 features no dataset de exemplo).
- Pipeline salvo como parte do `arvore_decisao_pipeline.joblib` após o GridSearch (se executado).

Observações

- O notebook tenta preservar nomes de features usando `get_feature_names_out` do OneHotEncoder; se não estiver disponível na versão da sua biblioteca, ajustamos a chamada automaticamente.