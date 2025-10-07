# 04 - Treinamento

Objetivo

- Treinar um `DecisionTreeClassifier` dentro de um `Pipeline` que inclui o `preprocessor`. Usar `GridSearchCV` com `StratifiedKFold` para otimização de hiperparâmetros.

Trechos de código relevantes

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=42))
])
param_grid = {
    'clf__criterion': ['gini','entropy'],
    'clf__max_depth': [3,5,7,9,None],
    'clf__min_samples_split': [2,5,10]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1, return_train_score=True)
grid.fit(X_train, y_train)
# Salvar melhor modelo
import joblib
joblib.dump(grid.best_estimator_, 'arvore_decisao_pipeline.joblib')
```

Arquivos gerados

- `grid_cv_results_arvore.csv` — resultados detalhados do GridSearch.
- `arvore_decisao_pipeline.joblib` — pipeline final salvo.

Resultados resumidos

- Melhores parâmetros e pontuação F1 por CV são salvos em `grid_cv_results_arvore.csv`. Para visualizar, abra o arquivo ou execute a célula correspondente no notebook.

Observações

- A execução completa do GridSearch pode demorar dependendo da máquina; o notebook está configurado para usar `n_jobs=-1` para paralelizar.