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

Arquivos gerados e interpretação dos resultados do GridSearch

O GridSearch grava uma tabela detalhada com métricas e tempos por combinação de hiperparâmetros. A seguir está uma explicação prática da estrutura do arquivo gerado (`grid_cv_results_arvore.csv`) e como interpretá-lo para escolher um modelo:

- Colunas principais encontradas no CSV e o que significam:
    - `mean_fit_time`, `std_fit_time`: tempo médio (e desvio) gasto para ajustar o pipeline em cada fold. Útil para estimar custo computacional.
    - `mean_score_time`, `std_score_time`: tempo médio para pontuar (predict/predict_proba) em cada fold.
    - `param_clf__...` / `params`: valores dos hiperparâmetros testados (por exemplo, `clf__criterion`, `clf__max_depth`, `clf__min_samples_split`). A coluna `params` costuma conter o dicionário completo da combinação.
    - `split0_test_score`, ..., `split4_test_score`: pontuações (aqui usamos F1) obtidas em cada fold do CV para a combinação; mostram variabilidade entre folds.
    - `mean_test_score`, `std_test_score`: média e desvio padrão das pontuações nos folds — a métrica principal para comparar combinações.
    - `rank_test_score`: posição ordenada por `mean_test_score` (1 = melhor média entre as combinações testadas).
    - `split0_train_score`, ..., `mean_train_score`, `std_train_score`: similares às colunas de teste, mas calculadas no conjunto de treino durante o CV; permitem avaliar overfitting quando muito maiores que as métricas de teste.

- Como interpretar as colunas na prática:
    1. Priorize combinações com `mean_test_score` alto e `std_test_score` baixo — indica boa média e baixa variabilidade entre folds.
    2. Compare `mean_train_score` x `mean_test_score`: um gap grande (treino muito maior que teste) sugere overfitting; prefira combinações com gap menor se a diferença de `mean_test_score` for pequena.
    3. Use `rank_test_score` para identificar os candidatos, mas sempre verifique `std_test_score` e o gap treino/teste antes de escolher a melhor combinação.
    4. Considere também `mean_fit_time` se houver restrição de tempo — modelos muito complexos podem ter ganhos marginais com custo alto.

Exemplo concreto (linha de exemplo extraída do CSV):

-- Parâmetros exemplo (melhor rank = 1): `{'clf__criterion': 'entropy', 'clf__max_depth': None, 'clf__min_samples_split': 10}`
     - `mean_test_score` = 0.3755 (média do F1 entre os 5 folds)
     - `std_test_score` = 0.0554 (variabilidade entre folds)
     - `mean_train_score` = 0.8513 (média F1 no treino — muito maior que no teste)
     - Interpretação: embora esta combinação tenha o melhor `mean_test_score` (rank 1), o gap entre treino e teste (~0.476) indica provável overfitting — o classificador se ajustou muito bem aos dados de treino mas generaliza pior.

Recomendações para seleção de hiperparâmetros a partir do CSV

1. Filtrar as top-k combinações por `mean_test_score` (por exemplo top 5) e inspecionar `std_test_score` e `mean_train_score` para escolher a combinação mais estável e generalizável.
2. Se a melhor combinação apresenta grande gap treino/teste, considerar uma combinação com `mean_test_score` levemente menor, porém com gap e `std_test_score` bem menores (mais robusta).
3. Preferir modelos mais simples (menor `max_depth`, maiores `min_samples_split`) quando a performance for similar — reduz complexidade e melhora interpretabilidade.
4. Documentar as combinações candidatas e testar a combinação final em um hold-out adicional (se possível) antes de promover o pipeline para produção.

Exemplo rápido de código (pandas) para inspecionar o CSV e escolher candidatos:

```python
import pandas as pd
df = pd.read_csv('grid_cv_results_arvore.csv')
# ordenar por média de teste decrescente
top = df.sort_values('mean_test_score', ascending=False).head(10)
print(top[['params','mean_test_score','std_test_score','mean_train_score']])
```

Conclusão

O CSV do GridSearch é a fonte definitiva para tomar decisões de hiperparâmetros: não escolha apenas pela melhor média sem checar estabilidade (desvio) e gap treino/teste. Use os critérios acima (média alta, baixo desvio, gap pequeno, custo computacional aceito) para selecionar a combinação final e, se necessário, reexecute um ajuste fino (por exemplo, restringindo a grade) nas regiões promissoras.

---

## 📊 Resumo Quantitativo da Busca

| Item | Valor |
|------|-------|
| Combinações de hiperparâmetros avaliadas | **30** |
| Folds de CV | **5** |
| Fits totais executados | **150** (30 × 5) |
| Melhor `mean_test_score` (F1) | **0.3755** |
| Mediana `mean_test_score` | **0.3491** |
| Média `mean_test_score` | **0.3431** |
| Desvio padrão global (`mean_test_score`) | **0.0280** |
| Menor `mean_test_score` | **0.2826** |
| Maior gap treino − teste | **0.6432** |
| Menor gap treino − teste | **0.0879** |
| Gap médio | **0.3610** |
| Gap mediano | **0.3907** |

Observação: o gap médio elevado indica tendência geral da árvore a sobreajustar quando a profundidade não é limitada ou quando `min_samples_split` é baixo.

## 🏅 Top 5 Combinações (ordenadas por `rank_test_score`)

| Rank | Critério | max_depth | min_samples_split | F1 (mean_test) | std_test | F1 treino (mean_train) | Gap |
|------|----------|-----------|-------------------|---------------|----------|------------------------|------|
| 1 | entropy | None | 10 | 0.3755 | 0.0554 | 0.8513 | 0.4758 |
| 2 | gini | 5 | 2 | 0.3699 | 0.0846 | 0.6277 | 0.2577 |
| 3 | entropy | 9 | 5 | 0.3691 | 0.0523 | 0.8818 | 0.5127 |
| 4 | gini | 5 | 10 | 0.3689 | 0.0840 | 0.6158 | 0.2469 |
| 5 | gini | 5 | 5 | 0.3687 | 0.0833 | 0.6224 | 0.2538 |

Notas rápidas:

- As posições 2, 4 e 5 (todas com `max_depth=5`) apresentam gaps bem menores com F1 quase igual ao topo.

- A combinação rank 1 tem melhor F1, porém gap muito alto (indicando sobreajuste potencial e menor interpretabilidade por árvore profunda ilimitada).

- A combinação rank 3 também sofre com gap alto (>0.5), reduzindo confiança na generalização.

## 🎯 Estabilidade vs Performance

- Entre as top 10 combinações, a de menor `std_test_score` (mais estável) tem `rank=6`, `entropy`, `max_depth=None`, `min_samples_split=2`, mas apresenta gap extremo (treino perfeito: 1.0) → forte sobreajuste.
- Para produção, priorizamos equilíbrio: F1 competitivo + gap moderado + variância aceitável.

### Candidatos Recomendados

| Justificativa | Combinação sugerida |
|---------------|---------------------|
| Menor gap entre as top 5 com F1 alto | `gini`, `max_depth=5`, `min_samples_split=10` |
| Trade-off entre F1 e estabilidade | `gini`, `max_depth=5`, `min_samples_split=5` |
| Se desejar ligeiro risco para ganho marginal | `gini`, `max_depth=5`, `min_samples_split=2` |

Recomenda-se revalidar as 3 candidatas acima em um hold-out (teste final ou validação temporal) e inspecionar métricas por classe (recall da classe minoritária especialmente).

## 🌳 Complexidade do Modelo Escolhido (Rank 1)

| Atributo | Valor |
|----------|-------|
| Critério | entropy |
| max_depth (param) | None (ilimitado) |
| min_samples_split | 10 |
| Profundidade real (`depth_actual`) | 13 |
| Número de nós (`node_count`) | 175 |
| Folhas | 88 |
| Razão folhas/nós | 0.5029 |

Comentários:
- Profundidade 13 para um dataset com ~1500 linhas é relativamente alta e facilita memorizar padrões raros (overfitting potencial).
- Razão ~0.50 indica estrutura relativamente equilibrada, mas o gap observado sugere que muitos caminhos estão refinando ruído.
- Um modelo com `max_depth=5` reduziria drasticamente complexidade e facilitaria interpretação de regras extraídas.

## 🔍 Interpretação do Gap

Gap (F1 treino − F1 validação) alto indica que a árvore explorou detalhes específicos dos folds de treino que não se repetem nos folds de validação. As principais causas aqui:

1. Profundidade ilimitada (crescimento até impureza mínima).

2. Divisões com poucos exemplos em folhas terminais.

3. Classe minoritária (Attrition=1) escassa, estimulando splits que segmentam poucos casos positivos.

Mitigações sugeridas:
- Fixar `max_depth=5–7`.
- Aumentar `min_samples_split` (já foi testado 10; avaliar 12–20).
- Aplicar `class_weight='balanced'` em nova rodada (não incluído na grade atual).
- Experimentar poda pós-treinamento (`ccp_alpha` via path de custo mínimo).

## 🧪 Próximos Passos Recomendados

1. Nova grade focalizada em torno de `max_depth ∈ {5,6,7}` e `min_samples_split ∈ {5,10,15}` adicionando `class_weight` e `ccp_alpha`.

2. Calibrar probabilidade (se for usar thresholds customizados) com `CalibratedClassifierCV` ou isotonic.

3. Comparar baseline com modelos alternativos simples (Logistic Regression, RandomForest shallow) para validar escolha.

4. Registrar versão do pipeline (hash + data) e congelar dependências (`requirements.txt`).

5. Gerar relatório de importância de features restrito às regras realmente utilizadas (via percurso da árvore podada).

## ✅ Checklist de Qualidade (Treinamento)

| Item | Status |
|------|--------|
| Reprodutibilidade (`random_state` em todos os componentes) | OK |
| Estratégia de validação estratificada | OK |
| Métrica coerente com classe desbalanceada (F1) | OK |
| Avaliação de estabilidade (`std_test_score`) | OK |
| Análise de overfitting (gap) | OK |
| Complexidade da árvore documentada | OK |
| Alternativas e próximos passos definidos | OK |
| Critérios de seleção explícitos | OK |

---

### Snippet para Reproduzir a Extração de Estatísticas

```python
import pandas as pd
df = pd.read_csv('grid_cv_results_arvore.csv')
df = df.sort_values('rank_test_score')
df['gap'] = df['mean_train_score'] - df['mean_test_score']
print(df[['rank_test_score','param_clf__criterion','param_clf__max_depth','param_clf__min_samples_split','mean_test_score','std_test_score','mean_train_score','gap']].head())
```

> Todos os números desta página foram extraídos automaticamente do CSV e do modelo salvo (`arvore_decisao_pipeline.joblib`)