---
hide:
- toc
---

# 02 - Pré-processamento (KNN)

Esta etapa transforma o dataset bruto em uma matriz de atributos adequada ao K-Nearest Neighbors, garantindo comparabilidade das escalas e representação não ordinal das variáveis categóricas.

## Objetivo

Preparar as features para que a métrica de distância (Manhattan, p=1) reflita relevância real entre registros, reduzindo viés de escala e de codificação.

## Visão Geral do Pipeline

| Etapa | Ação | Escopo | Justificativa |
|-------|------|--------|---------------|
| Seleção | Separar numéricas vs categóricas | 26 num / 9 cat | Definir transformações específicas |
| Imputação | Mediana (num) / Moda (cat) | Todas | Robustez a outliers / consistência valores faltantes (não havia missing, mas preserva robustez) |
| Codificação | One-Hot Encoding | Categóricas (exceto constante) | Evitar ordens artificiais |
| Escalonamento | StandardScaler | Numéricas | Distâncias comparáveis (KNN sensível a escala) |
| (Opcional futuro) | PCA / Seleção | Pós-transformação | Reduzir sparsity e ruído |

## Inventário de Variáveis

| Tipo | Quantidade | Observações |
|------|------------|-------------|
| Numéricas | 26 | Magnitudes heterogêneas (ex: renda vs anos) |
| Categóricas | 9 | Inclui `JobRole`, `EducationField`, `BusinessTravel` |
| Constantes | 1 | `Over18` (remoção recomendada) |
| Alvo | 1 | `Attrition` (não transformado) |

## Cardinalidade Categórica (Pré One-Hot)

| Coluna | Cardinalidade |
|--------|---------------|
| JobRole | 9 |
| EducationField | 6 |
| BusinessTravel | 3 |
| Department | 3 |
| MaritalStatus | 3 |
| Gender | 2 |
| OverTime | 2 |
| Attrition | 2 (alvo) |
| Over18 | 1 (constante) |

Total estimado de colunas dummies (excluindo alvo e constante): 9 + 6 + 3 + 3 + 3 + 2 + 2 = **28**.

## Dimensão Final Estimada

| Componente | Quantidade |
|------------|------------|
| Numéricas originais (escaladas) | 26 |
| Dummies geradas | 28 |
| Total estimado de features | **54** |

Nota: Valor estimado (não considerou possível remoção de redundâncias ou combinações futuras). 

## Estratégias Aplicadas

| Aspecto | Escolha | Justificativa Técnica |
|---------|--------|-----------------------|
| Imputação Numéricas | Mediana | Estável perante outliers, mesmo se surgirem em produção |
| Imputação Categóricas | Moda | Mantém distribuição original |
| Codificação | One-Hot (drop=None) | Evitar colinearidade interpretativa por enquanto; possível `drop='first'` em modelos lineares |
| Escalonamento | StandardScaler | Centra e normaliza variância para métrica p=1 |
| Alvo | Não transformado | Classificação binária direta |

## Justificativas Específicas

| Decisão | Alternativa Rejeitada | Motivo |
|---------|-----------------------|--------|
| StandardScaler | MinMaxScaler | Menos robusto a outliers; amplitude não crítica aqui |
| One-Hot completo | OrdinalEncoder | Evita induzir ordens inexistentes |
| Sem redução dimensional inicial | PCA imediato | Primeiro medir baseline para saber impacto real |
| Mediana imputação | Média | Média sensível a assimetrias (ex: renda) |

## Riscos e Mitigações

| Risco | Impacto no KNN | Mitigação Planejada |
|-------|----------------|---------------------|
| Alta dimensionalidade esparsa | Distâncias menos discriminativas | PCA / seleção por permutação |
| Variável constante incluída | Ruído irrelevante | Remover `Over18` antes de grid search ampliado |
| Desbalanceamento alvo | Recall baixo classe 1 | Ajuste de threshold / reamostragem | 
| Correlação entre dummies | Peso excessivo de grupos | Agrupar categorias raras / regularização via redução |

## Snippet do Pipeline (Exemplo)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_features = numeric_cols  # lista de 26 numéricas
categorical_features = [c for c in cat_cols if c not in ['Attrition','Over18']]

numeric_pipeline = Pipeline([
	("imputer", SimpleImputer(strategy="median")),
	("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
	("imputer", SimpleImputer(strategy="most_frequent")),
	("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
	("num", numeric_pipeline, numeric_features),
	("cat", categorical_pipeline, categorical_features)
])
```

## Checklist Pré-processamento

| Item | Status |
|------|--------|
| Tipos separados | OK |
| Imputação definida | OK |
| Escalonamento configurado | OK |
| Codificação categórica clara | OK |
| Remoção de constante mapeada | OK |
| Dimensão final estimada | OK |
| Riscos e mitigação anotados | OK |
| Snippet exemplar incluído | OK |

## Próximos Passos

1. Remover `Over18` e confirmar ausência de outras colunas com variância zero.

2. Experimentar PCA (buscando manter Macro F1 ou melhorar recall classe 1).

3. Rodar importância por permutação pós KNN para cortar dummies de baixo impacto.

4. Testar codificação alvo (Target Encoding) controlada para categorias de baixa cardinalidade e comparar.

5. Adicionar monitoramento de drift de cardinalidade em produção.

---

