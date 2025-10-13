---
hide:
- toc
---# 04 - Treinamento (KNN)



## Objetivo

Construir e avaliar um classificador K-Nearest Neighbors (KNN) para prever `Attrition`, utilizando os melhores hiperparâmetros obtidos (ou derivados do estudo de Elbow/Silhouette e validação) e documentar desempenho, limitações e próximos passos.

## Arquivos Relevantes

| Tipo | Arquivo | Descrição |
|------|---------|-----------|
| Modelo salvo | `knn_model.joblib` | Objeto treinado (possivelmente dentro de Pipeline) |
| Rótulos de teste | `knn_y_test.csv` | Coluna `Attrition` real (294 linhas) |
| Predições | `knn_y_pred.csv` | Coluna `y_pred` gerada pelo modelo |
| Notebook | `knn.ipynb` | Processo interativo de exploração/treino |

## Hiperparâmetros do Modelo Final

| Parâmetro | Valor |
|-----------|-------|
| `n_neighbors` | 3 |
| `weights` | uniform |
| `metric` | minkowski (p=1 → distância Manhattan) |
| `p` | 1 |
| `algorithm` | auto |
| `leaf_size` | 30 |

Observação: `p=1` indica uso da distância de Manhattan. Isso pode tornar o modelo mais robusto a outliers dimensionais comparado à distância Euclidiana (`p=2`).

## Distribuição da Classe no Conjunto de Teste

| Classe | Contagem | Proporção |
|--------|----------|-----------|
| 0 (Não Saiu) | 247 | 84.0% |
| 1 (Saiu) | 47 | 16.0% |
| Total | 294 | 100% |

Predições do modelo:

| Classe Predita | Contagem | Proporção |
|----------------|----------|-----------|
| 0 | 272 | 92.5% |
| 1 | 22 | 7.5% |

Deslocamento: o modelo reduziu a proporção de positivos previstos (de 16% reais para 7.5%), sinalizando viés para a classe majoritária.

## Métricas de Desempenho (Teste)

| Métrica | Classe 0 | Classe 1 | Macro Avg | Weighted Avg |
|---------|----------|----------|-----------|--------------|
| Precision | 0.8713 | 0.5455 | - | - |
| Recall | 0.9595 | 0.2553 | - | - |
| F1 | 0.9133 | 0.3478 | 0.6306 | 0.8229 |
| Suporte | 247 | 47 | 294 | 294 |
| Accuracy | \- | \- | \- | **0.8469** |

Interpretando:
- Alto recall (0.9595) para classe 0 e recall baixo (0.2553) para classe 1 → muitos falsos negativos (colaboradores que saíram, mas modelo previu que ficariam).
- F1 da classe 1 (0.3478) é o gargalo principal. Macro F1 (0.6306) reflete esse desequilíbrio de performance.
- Accuracy inflada pelo desbalanceamento (classe 0 domina 84%).

## Matriz de Confusão

| | Previsto 0 | Previsto 1 |
|---|-----------|-----------|
| Real 0 | 237 (TN) | 10 (FP) |
| Real 1 | 35 (FN) | 12 (TP) |

Indicadores derivados:
- Falsos Negativos (FN): 35 (74.5% dos 47 positivos reais não detectados)
- Falsos Positivos (FP): 10
- Taxa de detecção de positivos (sensibilidade classe 1): 12 / 47 = 25.5%

## Análise Crítica

Problemas observados:
1. Viés forte para classe 0 (subestima churn) → risco operacional (casos de saída despercebidos).
2. `n_neighbors=3` pode ser pequeno, favorecendo variação e sensibilidade a ruído local.
3. Distância Manhattan sem normalização adequada de escala (caso não tenha sido padronizado antes) pode distorcer relevância de features de maior amplitude.
4. Ausência de ajuste de limiar (usa probabilidade implícita por votação simples). Pode haver probabilidade útil acima de certo cutoff.

Hipóteses para baixo recall da classe 1:

- Classe minoritária dispersa no espaço vetorial → poucos vizinhos positivos próximos.

- Features categóricas codificadas (one-hot) com alta dimensionalidade diluindo densidade local.

- Falta de ponderação por distância (`weights='uniform'`).

## Melhorias Recomendadas

| Ação | Objetivo | Observação |
|------|----------|------------|
| Testar `weights='distance'` | Aumentar influência de vizinhos mais próximos | Pode melhorar recall classe 1 |
| Aumentar `n_neighbors` (5–15) | Reduzir variância, suavizar decisões | Monitorar queda em precision classe 1 |
| Rebalanceamento (SMOTE ou class weighting upstream) | Aumentar densidade de positivos | Aplicar antes do KNN com cuidado para overfitting sintético |
| Padronização/escala revisada | Garantir comparabilidade de atributos | Confirmar pipeline existente |
| Feature selection ou redução (PCA) | Mitigar sparsity pós one-hot | Rodar PCA preservando >90% variância |
| Ajuste de threshold via curva Precision-Recall | Melhorar trade-off recall/precision classe 1 | Requer probabilidades (usar `KNeighborsClassifier.predict_proba`) |
| Avaliar modelos alternativos (LogReg, Árvores Podadas) | Benchmark | Focar em recall classe 1 |

## KPIs Prioritários (Churn)

| KPI | Valor Atual | Meta Sugerida (Iteração Próxima) |
|-----|-------------|----------------------------------|
| Recall classe 1 | 0.2553 | ≥ 0.45 |
| Precision classe 1 | 0.5455 | ≥ 0.50 (manter) |
| F1 classe 1 | 0.3478 | ≥ 0.48 |
| Macro F1 | 0.6306 | ≥ 0.70 |

## Checklist de Qualidade

| Item | Status |
|------|--------|
| Hiperparâmetros finais documentados | OK |
| Distribuição real vs predita analisada | OK |
| Métricas por classe incluídas | OK |
| Matriz de confusão explicada | OK |
| Identificação de viés (classe 1) | OK |
| Recomendações de tuning propostas | OK |
| KPIs definidos para próxima iteração | OK |
| Próximos passos claros | OK |

## Próximos Passos Sugeridos

1. Rodar nova busca com grade: `n_neighbors ∈ {5,7,9,11,13}`, `weights ∈ {uniform,distance}`, incluir validação estratificada.
2. Gerar curva Precision-Recall e definir threshold maximizando F1 ou recall condicionado a precision mínima (ex: >=0.5).
3. Testar pipeline com PCA (ex: 0.95 variância explicada) antes do KNN.
4. Comparar recall classe 1 com um classificador linear regularizado.
5. Registrar experimentos (ex: MLflow ou planilha de tracking) para cada iteração.

---

### Snippet para Reproduzir Avaliação

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

y_true = pd.read_csv('knn_y_test.csv')['Attrition']
y_pred = pd.read_csv('knn_y_pred.csv')['y_pred']
print(classification_report(y_true, y_pred, digits=4))
print(confusion_matrix(y_true, y_pred))
```

> Números extraídos automaticamente em 07/10/2025. TN=237, FP=10, FN=35, TP=12.
