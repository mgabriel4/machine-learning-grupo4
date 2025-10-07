---
title: Relatório Final - Machine Learning Grupo 4
---

# Relatório Final

Este documento consolida o trabalho realizado no projeto: Análise exploratória, pré-processamento, treinamento e avaliação de modelos (Árvore de Decisão, KNN e K-Means). Inclui métricas principais, visualizações geradas e recomendações de melhoria.

## 1. Resumo executivo

- Modelos implementados: Árvore de Decisão, K-Nearest Neighbors (KNN), K-Means (clustering).
- Dados: `funcionarios.csv` (subset usado nos notebooks em `docs/*/`).
- Entregáveis: Notebooks Jupyter, modelos salvos (`.joblib`), CSVs de predições e várias imagens em `docs/*/imagens/`.

Resumo da performance (pontos principais):

- KNN: GridSearchCV executado; melhor combinação encontrada: n_neighbors=3, p=1 (Manhattan), weights=uniform. Best CV F1 ≈ 0.28. Modelo salvo em `docs/knn/knn_model.joblib`.
- Decision Tree: pipeline montado, árvore treinada, visualização completa e árvore reduzida para apresentação disponível em `docs/arvore_decisao/`.
- K-Means: elbow e silhouette analisados; artefatos e visualizações em `docs/kmeans/`.

## 2. Metodologia

Fluxo padrão seguido para cada modelo:

1. Carregamento e EDA: investigação de distribuições, contagens e estatísticas descritivas.
2. Pré-processamento: imputação (mediana para numéricas, moda para categóricas), codificação One-Hot para categóricas (`handle_unknown='ignore'`), e escalonamento (StandardScaler) quando aplicável.
3. Divisão treino/teste: `train_test_split(..., stratify=y)` quando o alvo está presente.
4. Treinamento: GridSearchCV com StratifiedKFold (onde aplicável), escolha do melhor modelo por métrica F1 para classificadores.
5. Avaliação: `classification_report`, matriz de confusão, ROC/PR quando disponível; para K-Means, inertia e silhouette.

## 3. Resultados detalhados por modelo

### 3.1 K-Nearest Neighbors (KNN)

- Melhor configuração (CV): n_neighbors=3, p=1, weights='uniform'.
- Métricas no conjunto de teste:

```text
Accuracy: (veja arquivo `docs/knn/knn_y_test.csv` e `docs/knn/knn_y_pred.csv`)
F1 (macro/weighted): ver classification_report no notebook `docs/knn/knn.ipynb`
Best CV F1 (GridSearch): ~0.2798
```

Arquivos importantes:

- Modelo: `docs/knn/knn_model.joblib`
- Previsões: `docs/knn/knn_y_pred.csv` e `docs/knn/knn_y_test.csv`
- Imagens: `docs/knn/imagens/knn_confusion_matrix.png`, `knn_roc.png`, `knn_pr_curve.png`, `knn_pca2d.png`, `knn_correlation.png`, `knn_dist_*.png`

Observações: F1 relativamente baixo indica necessidade de engenharia de features e/ou balanceamento de classes.

### 3.2 Árvore de Decisão

- Pipeline com imputação, codificação e treinamento via GridSearch (detalhes no notebook `docs/arvore_decisao/arvore_decisao.ipynb`).
- Artefatos:
  - Notebook completo e imagem da árvore reduzida para apresentação em `docs/arvore_decisao/imagens/`.
  - Feature importances calculadas e salvadas no notebook.

### 3.3 K-Means (Clustering)

- Elbow e Silhouette analisados para k entre 2 e 10; artefatos salvos em `docs/kmeans/imagens/`.
- Resultado final experimentado: k=2 (baseado nas análises), com clusters salvos em `docs/kmeans/kmeans_clusters.csv` e visualização em `docs/kmeans/imagens/kmeans_clusters.png`.

## 4. Comparação e Tabela resumida

> Observação: Os números detalhados estão nos notebooks e CSVs; abaixo um resumo qualitativo e as métricas chave já calculadas (ver notebooks para tabelas completas).

| Modelo | Tipo | Métrica alvo | Observação |
|---|---:|---:|---|
| Árvore de Decisão | Classificação | F1 / Acurácia | Robustez e interpretabilidade; ver notebook para melhores hyperparâmetros |
| KNN | Classificação | F1 ≈ 0.28 (CV) | Bom baseline; pode melhorar com balanceamento/engenharia |
| K-Means | Clustering | Silhouette (máximo ≈ 0.154) | Clusters fracos — pouca separação clara nos dados numéricos usados |

## 5. Visualizações principais

As principais imagens estão em:

- `docs/knn/imagens/` — confusion matrix, ROC/PR, PCA 2D, correlação, distribuições.
- `docs/arvore_decisao/imagens/` — árvore completa e reduzida, feature importances.
- `docs/kmeans/imagens/` — elbow, silhouette e clusters em PCA.

(Estas imagens foram incorporadas às páginas de cada experimento no diretório `docs/`.)

## 5.1 Métricas numéricas (extraídas dos artefatos)

As métricas abaixo foram calculadas automaticamente a partir dos artefatos gerados pelo fluxo (arquivos CSV e modelos salvos).

### KNN (avaliação no conjunto de teste)

| Métrica | Valor |
|---|---:|
| Accuracy | 0.8469 |
| Precision (macro) | 0.7084 |
| Recall (macro) | 0.6074 |
| F1 (macro) | 0.6306 |
| Precision (weighted) | 0.8192 |
| Recall (weighted) | 0.8469 |
| F1 (weighted) | 0.8229 |
| Support (n_test) | 294 |

Per-class (trecho do classification_report):

- Classe 0 (No): precision=0.8713, recall=0.9595, f1=0.9133, support=247
- Classe 1 (Yes): precision=0.5455, recall=0.2553, f1=0.3478, support=47

### KMeans

| Métrica | Valor |
|---|---:|
| Silhouette (clusters usados) | 0.1869 |


## 6. Conclusões e Recomendações

- O pipeline está implementado e reprodutível: notebooks, modelos e artefatos estão salvos.
- Pontos fortes:
  - Uso consistente de Pipeline/ColumnTransformer, que facilita reprodutibilidade e integração.
  - Boas visualizações para comunicar resultados (árvore reduzida, plots KMeans, ROC/PR).

- Pontos de melhoria (prioridade):
  1. Balanceamento das classes no KNN (SMOTE ou undersampling) e re-treino para melhorar F1.
  2. Testar modelos robustos (RandomForest, XGBoost) com feature selection/importance comparada.
  3. Documentar um relatório comparativo final (esta página) com tabelas numéricas extraídas automaticamente de `cv_results_` e dos testes.
  4. Incluir explicitação das razões de exclusão/inclusão de features no pré-processamento (justificativas de negócio).

## 7. Como reproduzir (resumo rápido)

1. Ative o ambiente Python com as dependências listadas em `requirements.txt`.
2. Abra os notebooks em `docs/arvore_decisao/`, `docs/knn/` e `docs/kmeans/`.
3. Execute as células na ordem (1 → imports → load → preprocess → train → evaluate). Imagens e artefatos serão salvos nas pastas `imagens/` e nos arquivos `.joblib`/`.csv` correspondentes.

## 8. Próximos passos sugeridos

1. Gerar um relatório comparativo completo em forma de tabela (podemos automatizar isso).  
2. Implementar balanceamento e testar RandomForest/XGBoost para melhorar F1.  
3. Extrair `cv_results_` e anexar os melhores experimentos na pasta `docs/relatorio_final/assets/`.

---

Relatório gerado automaticamente a partir dos notebooks e artefatos no repositório.
