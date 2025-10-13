---
hide:
- toc
---

# 03 - Seleção de Hiperparâmetros (KNN)

Este documento descreve a busca de hiperparâmetros do KNN, critérios de avaliação e interpretação dos resultados obtidos via validação cruzada estratificada.

## Objetivo

Encontrar combinação de `n_neighbors`, `weights` e métrica de distância (`p`) que maximize o F1 (classe minoria incluída no cálculo macro) preservando generalização.

## Métrica Primária

- Optamos por F1 (macro) em vez de Accuracy para penalizar o baixo recall da classe positiva (`Attrition = Yes`) dado o desbalanceamento (~16%).

## Espaço de Busca

| Hiperparâmetro | Valores Avaliados | Observação |
|----------------|-------------------|------------|
| n_neighbors | 3, 5, 7, 9 | Menores valores capturam fronteiras locais; maiores suavizam ruído |
| weights | uniform, distance | `distance` pode ajudar quando densidade local é heterogênea |
| p (Minkowski) | 1 (Manhattan), 2 (Euclidiana) | p=1 mais robusto a outliers em algumas dimensões |

Total de combinações: 4 × 2 × 2 = 16.

## Metodologia

| Aspecto | Decisão | Justificativa |
|---------|---------|---------------|
| Validação | StratifiedKFold (5 folds) | Mantém proporção da classe minoritária em cada fold |
| Shuffle | Ativado (padrão scikit) | Reduz viés de ordenação original |
| Métrica GridSearch | F1 (macro) | Equilíbrio entre precisão e recall em ambas as classes |
| Repetições | Única passagem | Custo computacional baixo dado espaço pequeno |
| Paralelismo | n_jobs adequado (não documentado) | Tempo de busca reduzido |

## Resultados Principais

| Combinação Ótima | F1 Médio (CV) | Observações |
|------------------|--------------|-------------|
| n_neighbors=3, weights=uniform, p=1 | 0.27975 | Melhor trade-off entre precisão classe 0 e recall classe 1 nesta malha inicial |

Resumo adicional qualitativo (inferido):

- `n_neighbors` maior (≥7) tende a diluir instâncias positivas raras, reduzindo recall.
- `weights='distance'` não superou `uniform` possivelmente por sparsity moderada e pouca sobreposição de densidades relevantes.
- Métrica p=1 favoreceu leve ganho por reduzir influência de grandes diferenças em dimensões escaladas.

## Interpretação

| Fator | Impacto | Explicação |
|-------|---------|------------|
| Baixo F1 macro (≈0.28) | Sinaliza dificuldade em capturar padrões da classe positiva | Fronteira possivelmente complexa e/ou necessidade de novas features |
| n_neighbors pequeno (3) | Modelo mais sensível a ruído | Mas melhora chance de capturar bolsões da classe minoritária |
| Falha de `distance` weighting | Distâncias já normalizadas; vizinhos próximos não diferem tanto | Peso inverso não gera discriminação adicional |

## Limitações da Busca Atual

| Limitação | Consequência | Próxima Ação |
|-----------|--------------|--------------|
| Espaço pequeno | Pode ter perdido combinações úteis (ex: n_neighbors=11) | Ampliar grade incremental |
| Ausência de técnicas de reamostragem | Recall baixo classe 1 | Integrar SMOTE ou Class Weight sintético via KNN adaptado |
| Sem ajuste de threshold | Métrica fixa no padrão 0.5 | Avaliar curva Precision-Recall para threshold ótimo |
| Sem seleção de features antes | Potencial ruído | Testar redução (PCA ou seleção por importância) |

## Recomendações

1. Expandir `n_neighbors` para faixa 3–25 com passo 2.

2. Incluir métrica p=1 e p=2, e testar p fracionário (ex: 1.5) via métrica custom (se viável).

3. Incorporar `weights='distance'` novamente após redução dimensional (efeito pode mudar).

4. Avaliar pipeline com PCA (ex: variância 90–95%) e repetir GridSearch.

5. Rodar análise de threshold utilizando curva Precision-Recall visando maximizar F1 da classe positiva.

6. Considerar modelos alternativos: Gradient Boosting, Random Forest, Logistic com regularização.

7. Testar reamostragem (SMOTE) em pipeline isolado para comparar recall classe 1.

## Checklist

| Item | Status |
|------|--------|
| Objetivo definido | OK |
| Métrica primária justificada | OK |
| Espaço de busca documentado | OK |
| Metodologia clara | OK |
| Resultado ótimo registrado | OK |
| Limitações apontadas | OK |
| Recomendações futuras listadas | OK |

## Próximos Passos Curto Prazo

- Rodar expansão de grade (n_neighbors até 25).
- Gerar tabela de comparação macro F1 vs recall positivo.
- Plotar curva F1 vs n_neighbors para cada combinação de `weights`.

## Registro

> Melhor combinação encontrada: n_neighbors=3, p=1, weights='uniform' (CV F1 macro = 0.27975). Atualizado em 07/10/2025.

---

Se métricas subsequentes (ver `04_treinamento.md`) divergirem, revisar consistência de seed, folds ou transformações.
