# 04 - Treinamento Final (K-Means)

Documento de consolidação do modelo K-Means na configuração base (K=2) com análise dos clusters obtidos.

## Objetivo

Treinar e persistir o modelo baseline para avaliação exploratória de segmentos, entendendo diferenças estruturais antes de incluir engenharia adicional (categóricas, PCA, novos K).

## Configuração do Modelo

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| K (n_clusters) | 2 | Silhouette máximo (ainda baixo) e simplicidade interpretativa inicial |
| init | k-means++ | Convergência mais estável |
| n_init | 10 | Reduz variância por inicialização |
| random_state | 42 | Reprodutibilidade |
| Escala | StandardScaler | Distâncias comparáveis |

## Artefatos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `kmeans_clusters.csv` | Rótulos de cluster (coluna `cluster`) |
| `kmeans_X.csv` | Matriz de features numéricas pós limpeza |
| `kmeans_model.joblib` | Objeto KMeans serializado |

## Código Base

```python
from sklearn.cluster import KMeans
import joblib, pandas as pd

K_FINAL = 2
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
labels_final = km_final.fit_predict(X_scaled)

pd.DataFrame(labels_final, columns=['cluster']).to_csv('kmeans_clusters.csv', index=False)
pd.DataFrame(X, columns=num_cols).to_csv('kmeans_X.csv', index=False)
joblib.dump(km_final, 'kmeans_model.joblib')
```

## Distribuição dos Clusters

| Cluster | Tamanho | % |
|---------|---------|----|
| 0 | 459 | 31.22 |
| 1 | 1011 | 68.78 |

Desequilíbrio moderado: cluster 1 representa maioria dos registros.

## Principais Diferenças Entre Clusters (z-score)

Top 10 variáveis que mais diferenciam os centróides (|Δ z|):

| Variável | |Δ z| |
|----------|-------|
| JobLevel | 1.50 |
| TotalWorkingYears | 1.49 |
| MonthlyIncome | 1.45 |
| YearsAtCompany | 1.40 |
| YearsInCurrentRole | 1.37 |
| YearsWithCurrManager | 1.29 |
| YearsSinceLastPromotion | 1.18 |
| Age | 0.95 |
| Education | 0.22 |
| StockOptionLevel | 0.10 |

Interpretação preliminar:
- Cluster 1 (maior) possivelmente agrega colaboradores em estágios mais iniciais / menor senioridade.
- Cluster 0 aparenta agrupar perfis com maior senioridade, tempo de casa, progressão e remuneração.

## Leitura de Negócio (Hipóteses)

| Aspecto | Hipótese | Ação Validar |
|---------|----------|--------------|
| Senioridade | Cluster 0 = colaboradores mais experientes | Cruzar com `Attrition` para ver se retenção difere |
| Progressão | Diferenças em `YearsSinceLastPromotion` | Ver turnover vs estímulo de promoção |
| Remuneração | `MonthlyIncome` mais alto em um cluster | Analisar se correlaciona com menor saída |

## Limitações do Modelo Atual

| Limitação | Impacto | Mitigação Próxima |
|-----------|---------|------------------|
| Baixo Silhouette geral | Segmentação fraca | Incluir categóricas + PCA |
| Apenas K=2 avaliado aqui | Perda de nuance | Comparar K=3 (DB menor) |
| Sem pureza de Attrition | Pouca validação de relevância | Calcular distribuição de `Attrition` por cluster |
| Escala reprocessada ad hoc | Reprodutibilidade parcial | Persistir scaler em joblib |

## Recomendações Próximas

1. Calcular tabela de médias por cluster (todas as features) e salvar como `kmeans_cluster_profile.csv`.
2. Incluir `JobRole`, `OverTime` e `MaritalStatus` (One-Hot) para nova rodada de K.
3. Testar K=3 e K=4 e comparar Silhouette + Davies-Bouldin.
4. Avaliar PCA (manter 90–95% variância) e repetir clustering para ver se Silhouette melhora.
5. Medir pureza de `Attrition` e gerar gráfico de barras por cluster.
6. Validar estabilidade com múltiplos `n_init` (ex: 30) e calcular desvio padrão de inertia.

## Checklist

| Item | Status |
|------|--------|
| Modelo treinado | OK |
| Artefatos salvos | OK |
| Distribuição clusters documentada | OK |
| Diferenças principais analisadas | OK |
| Limitações listadas | OK |
| Próximos passos definidos | OK |

## Observação Final

Esse baseline serve como referência; melhorias esperadas virão de engenharia de features e reavaliação de K.

---

> Atualizado em 07/10/2025.
