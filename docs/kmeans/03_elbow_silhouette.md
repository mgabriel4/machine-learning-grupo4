---
hide:
- toc
---

# 03 - Elbow & Silhouette (Seleção de K)

Avaliação interna do número de clusters usando Inertia (elbow), Silhouette e complemento com Davies-Bouldin.

## Objetivo

Determinar um intervalo razoável de K que balanceie redução de inércia e coesão/separação segundo Silhouette, evitando overfitting com muitos clusters de ganho marginal.

## Metodologia

| Aspecto | Decisão | Justificativa |
|---------|---------|---------------|
| Faixa K | 2 a 10 | Ponto inicial amplo sem explodir custo |
| Inicialização | k-means++ | Convergência mais estável |
| n_init | 10 | Reduz risco de mínimo local ruim |
| Métricas | Inertia, Silhouette, Davies-Bouldin | Complementar coesão vs separação |
| Escala | StandardScaler aplicado | Evitar dominância de variáveis de maior variância |

## Resultados (k=2..10)

| K | Inertia | Δ Inertia Relativa | Silhouette | Davies-Bouldin |
|---|---------|--------------------|------------|----------------|
| 2 | 29,253.48 | - | 0.154 | 2.374 |
| 3 | 27,407.52 | 6.31% | 0.132 | 2.292 |
| 4 | 25,979.56 | 5.21% | 0.096 | 2.581 |
| 5 | 24,915.10 | 4.10% | 0.103 | 2.418 |
| 6 | 24,089.73 | 3.31% | 0.078 | 2.658 |
| 7 | 23,671.70 | 1.74% | 0.079 | 2.667 |
| 8 | 23,296.42 | 1.59% | 0.065 | 2.785 |
| 9 | 22,931.69 | 1.57% | 0.062 | 2.753 |
| 10 | 22,670.96 | 1.14% | 0.061 | 2.945 |

Notas:
- Queda relativa da Inertia perde força a partir de K≥6 (<3.5%).
- Silhouette máximo em K=2 (0.154) ainda baixo (separação fraca global).
- Davies-Bouldin menor em K=3 (2.29), indicando leve melhor equilíbrio intra/inter.

## Interpretação

| Aspecto | Observação | Implicação |
|---------|------------|------------|
| Elbow difuso | Não há inflexão forte | Estrutura fraca/gradual |
| Silhouette baixo (<0.2) | Clusters parcialmente sobrepostos | Variáveis insuficientes ou não discriminativas |
| K=2 vs K=3 | Trade-off: Silhouette cai, mas DB melhora | K=3 pode capturar subgrupo adicional sem ganho claro de coesão |
| Ganhos marginais após K=5 | Δ inertia <5% | Evitar grande K por interpretabilidade |

## Limitações Atuais

| Limitação | Consequência | Próxima Ação |
|-----------|--------------|--------------|
| Só variáveis numéricas | Perda de semântica categórica | Incluir One-Hot de JobRole/OverTime |
| Sem redução dimensional | Possível redundância | Avaliar PCA (variância 90–95%) |
| Sem análise de estabilidade | Sensível a seeds | Rodar múltiplos n_init e comparar variação |
| Não avaliado contra atributo de negócio | Baixa interpretabilidade | Calcular pureza Attrition por cluster |

## Recomendações

1. Testar K=2, 3 e 4 após incluir subset de categóricas codificadas.
2. Calcular Silhouette por ponto e analisar distribuição (boxplot) para detectar outliers clusterizados.
3. Rodar PCA (2D/3D) e visualizar clusters para avaliar agrupamentos latentes.
4. Medir pureza de `Attrition` em cada cluster (sem usar no ajuste) para interpretar perfis.
5. Reexecutar com `RobustScaler` se variáveis com outliers forem confirmadas.
6. Adicionar métrica Calinski-Harabasz para reforçar comparação.

## Checklist

| Item | Status |
|------|--------|
| Faixa de K avaliada | OK |
| Métricas múltiplas | OK |
| Elbow interpretado | OK |
| Silhouette analisado | OK |
| Davies-Bouldin incluído | OK |
| Recomendações futuras | OK |
| Limitações documentadas | OK |

## Código Base (Extrato)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

inertias, silhouettes, db_scores = [], [], []
K_range = range(2, 11)
prev = None
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
```

## Observação Final

Os valores de Silhouette indicam que melhorias substanciais dependem de engenharia de features (inclusão categórica e/ou redução dimensional). Selecionar K somente com base nestes números levaria a solução de baixo valor de negócio.

---

> Atualizado em 07/10/2025.
