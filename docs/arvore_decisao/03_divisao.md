# 03 - Divisão treino / teste

Objetivo

- Separar os dados em conjuntos de treino e teste de forma estratificada para preservar a proporção da classe alvo.

Trechos de código relevantes

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)
```

Arquivos gerados

- `X_train_arvore.csv`, `X_test_arvore.csv`, `y_train_arvore.csv`, `y_test_arvore.csv` (o notebook contém células para salvar esses arquivos quando executado).

Resultados resumidos

- Distribuição da classe preservada entre treino e teste.
- Tamanho do teste: 20% do conjunto original.
