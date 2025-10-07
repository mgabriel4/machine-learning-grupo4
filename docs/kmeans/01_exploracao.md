# 01 - Exploração dos Dados (K-Means)

Esta etapa estabelece a linha de base estrutural do dataset antes da construção do pipeline de clustering (K-Means).

## Resumo Rápido

| Item | Valor |
|------|-------|
| Linhas | 1470 |
| Colunas | 35 |
| Numéricas | 26 |
| Categóricas | 9 |
| Missing (total) | 0 |
| Target `Attrition` (No / Yes) | 83.88% / 16.12% |
| Cardinalidade Máx (JobRole) | 9 |
| Coluna Constante | EmployeeCount (=1), Over18 (="Y") |
| Pós-processamento (`kmeans_X.csv`) | 1470 × 23 |
| Clusters detectados (atual) | 2 (0: 459 / 1: 1011) |

## Objetivos de Exploração

1. Verificar integridade (missing, constantes, cardinalidade).

2. Identificar potenciais variáveis redundantes ou pouco informativas para clusterização.

3. Avaliar necessidade de normalização, codificação e redução dimensional.

4. Entender distribuição do alvo (usado aqui apenas como variável de referência qualitativa – não para treinar K-Means supervisionado).

## Estrutura de Dados

| Tipo | Contagem | Observação |
|------|----------|------------|
| Numéricas (int/contínuas) | 26 | Inclui variáveis de avaliação, tempo, renda, taxas |
| Categóricas | 9 | Diversas escalas nominais (JobRole, Department, etc.) |
| Constantes | 2 | `EmployeeCount`, `Over18` – candidatas à remoção |

## Distribuição do Alvo (Referência)

Embora K-Means seja não supervisionado, manter a coluna `Attrition` para avaliação pós-cluster ajuda a medir alinhamento de grupos a um outcome de negócio.

| Classe | Contagem | % |
|--------|----------|----|
| No | 1233 | 83.88 |
| Yes | 237 | 16.12 |

Desequilíbrio relevante: clusters podem ser usados para segmentação de risco; não confundir com classificação direta.

## Cardinalidade de Categóricas

| Coluna | Cardinalidade |
|--------|---------------|
| JobRole | 9 |
| EducationField | 6 |
| BusinessTravel | 3 |
| Department | 3 |
| MaritalStatus | 3 |
| Gender | 2 |
| OverTime | 2 |
| Attrition | 2 |
| Over18 | 1 (constante) |

Implicação: One-Hot Encoding direto expande dimensionalidade; convém tratar colunas com alta cardinalidade (ex: `JobRole`) avaliando impacto em inércia.

## Amostra de Estatísticas Numéricas

| Variável | Média | Desvio | Min | Max |
|----------|-------|--------|-----|-----|
| Age | 36.92 | 9.14 | 18 | 60 |
| DailyRate | 802.49 | 403.51 | 102 | 1499 |
| DistanceFromHome | 9.19 | 8.11 | 1 | 29 |
| Education | 2.91 | 1.02 | 1 | 5 |
| EmployeeNumber | 1024.87 | 602.02 | 1 | 2068 |
| EnvironmentSatisfaction | 2.72 | 1.09 | 1 | 4 |
| HourlyRate | 65.89 | 20.33 | 30 | 100 |
| JobLevel | 2.06 | 1.11 | 1 | 5 |

Observações:
- Presença de escalas heterogêneas (ex: `EmployeeNumber` vs `JobLevel`) impõe normalização/ padronização (senão distâncias serão distorcidas).
- Variáveis claramente identificadoras ou administrativas (`EmployeeNumber`) provavelmente não agregam valor sem transformação.

## Pré-Processamento Atual (Resumo do Arquivo `kmeans_X.csv`)

| Aspecto | Situação |
|---------|----------|
| Linhas processadas | 1470 |
| Colunas finais | 23 |
| Redução de dimensões explícita | Não |
| Remoção de constantes | (Verificar se aplicada) |
| Escalonamento | (Assumir necessário antes de K-Means) |

## Distribuição de Clusters (Atual)

| Cluster | Tamanho | % |
|---------|---------|----|
| 0 | 459 | 31.22 |
| 1 | 1011 | 68.78 |

Desequilíbrio moderado. Próxima análise: verificar se um cluster está simplesmente espelhando volume da classe majoritária de Attrition (validar pureza).

## Qualidade e Potenciais Ajustes

| Item | Situação | Ação Recomendada |
|------|----------|------------------|
| Missing | 0 | OK |
| Constantes | Encontradas | Remover `EmployeeCount`, `Over18` |
| Escalas divergentes | Sim | Padronizar / normalizar |
| Alta cardinalidade | JobRole (9) | Avaliar agrupamento semântico |
| Ruído identificador | EmployeeNumber | Excluir |
| Potencial redundância | Education vs EducationField | Checar correlação pós encoding |

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Alta dimensionalidade pós One-Hot | Dilui separação | PCA / seleção variância |
| Clusters dominados por escala salarial | Viés interpretativo | Escalonar + analisar importâncias por variância |
| Componentes interpretativamente fracos | Adoção baixa | Produzir dicionário de variáveis |
| Mistura de variáveis administrativas | Distâncias artificiais | Remover IDs / constantes |

## Recomendações Próximas

1. Confirmar pipeline de normalização (StandardScaler ou RobustScaler).

2. Gerar curva Elbow (k=2..10) + Silhouette para recalibrar número de clusters.

3. Calcular métricas internas: Inertia, Silhouette, Davies-Bouldin.

4. Avaliar clusterização em espaço reduzido (PCA 2D / 3D) comparando separabilidade.

5. Medir pureza de Attrition por cluster (sem usar no treinamento, apenas validação).

6. Criar versão filtrada removendo variáveis administrativas para comparar inércia relativa.

7. Testar escalonamento robusto se outliers forem detectados em renda/ taxas.

## Checklist

| Item | Status |
|------|--------|
| Shape verificado | OK |
| Missing inspecionado | OK |
| Constantes identificadas | OK |
| Cardinalidades listadas | OK |
| Estatísticas numéricas | OK |
| Distribuição clusters | OK |
| Recomendações futuras | OK |

## Código Base (Exemplo de Extração)

```python
import pandas as pd
df = pd.read_csv('funcionarios.csv')
print(df.shape)
print(df.select_dtypes(include='number').describe().T[['mean','std','min','max']])
cat_cols = df.select_dtypes(include='object').columns
print({c: df[c].nunique() for c in cat_cols})
```

