# 03 - GridSearch e seleção de hiperparâmetros

Realizamos um GridSearchCV com StratifiedKFold (5 folds) para otimizar os hiperparâmetros do KNN:

- Parâmetros testados: n_neighbors = [3,5,7,9], weights = ['uniform','distance'], p = [1,2]
- Melhor combinação encontrada: n_neighbors=3, p=1, weights='uniform'
- Melhor F1 médio em CV: 0.27975

Observação: o F1 é baixo — recomenda-se testar outras arquiteturas ou engenharia de features.
