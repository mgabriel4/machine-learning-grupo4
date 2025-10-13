---
hide:
- toc
---

# 02 - Pré-processamento (K-Means)

Esta etapa descreve como o conjunto de atributos foi preparado antes da aplicação do algoritmo K-Means.

## Objetivo

Transformar o dataset bruto (1470 × 35) em uma matriz numérica coerente (1470 × 23) removendo colunas irrelevantes, imputando valores ausentes (não houve missing nesta base, mas estratégia definida), padronizando escalas e evitando variáveis identificadoras que distorçam distâncias.

## Inventário de Transformações

| Etapa | Ação | Escopo | Motivo |
|-------|------|--------|--------|
| Seleção Numéricas | Filtrar apenas colunas numéricas | 26 originais | K-Means requer dados contínuos (distâncias em espaço vetorial) |
| Remoção IDs/Constantes | Excluir `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18` | Identificadores/constantes | Não agregam estrutura; distorcem centroides |
| Imputação | Mediana (fallback) | (Sem missing) | Definida para robustez futura |
| Escalonamento | StandardScaler | Todas numéricas finais | Evitar dominância de atributos de grande variância |
| Persistência | Salvar `kmeans_X.csv` | 1470 × 23 | Reprodutibilidade |

## Colunas Removidas / Justificativa

| Coluna | Tipo | Razão de Remoção |
|--------|------|------------------|
| EmployeeNumber | Identificador | Não representa comportamento ou perfil |
| EmployeeCount | Constante (=1) | Zero variância |
| StandardHours | Constante (=80?) | Zero variância (assumido padrão) |
| Over18 | Constante ("Y") | Zero variância |

## Forma Antes vs Depois

| Fase | Shape |
|------|-------|
| Bruto | 1470 × 35 |
| Após seleção/remover IDs | 1470 × 23 |

Redução remove ruído sem introduzir sparsity.

## Considerações sobre Escalonamento

Escolhido `StandardScaler` (média 0, desvio 1) para aproximar isotropia das variáveis, já que K-Means minimiza soma das distâncias quadráticas aos centroides (função de inércia sensível à escala). Alternativas:

| Alternativa | Prós | Contras | Status |
|-------------|------|--------|--------|
| MinMaxScaler | Mantém amplitudes 0–1 | Sensível a outliers | Rejeitada |
| RobustScaler | Resistente a outliers | Pode achatar variáveis já estáveis | Avaliar se renda/taxas tiverem outliers extremos |
| Sem Escala | Simplicidade | Distâncias dominadas por magnitude | Rejeitada |

## Snippet do Pipeline (Versão Ilustrativa)

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('funcionarios.csv')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
drop_cols = ['EmployeeNumber','EmployeeCount','StandardHours']
num_cols = [c for c in num_cols if c not in drop_cols]
X = df[num_cols].copy()

# imputação (sem missing atual, mas definido)
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pd.DataFrame(X, columns=num_cols).to_csv('kmeans_X.csv', index=False)
```

## Qualidade e Integridade

| Verificação | Resultado | Ação |
|-------------|-----------|------|
| Missing | 0 | OK |
| Constantes | Removidas | OK |
| IDs | Removidos | OK |
| Escala homogênea | Após scaler | OK |
| Duplicatas (linhas) | Não avaliado | Verificar na próxima iteração |

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Perda de informação categórica | Clusters pouco interpretáveis | Incluir dummies selecionadas em versão 2 |
| Variáveis altamente correlacionadas | Centroides redundantes | PCA opcional após avaliação de inércia |
| Outliers não tratados | Centroides deslocados | Analisar boxplots e aplicar RobustScaler se necessário |
| Escala re-aprendida sem persistir scaler | Reprodutibilidade comprometida | Salvar objeto scaler (joblib) |

## Próximas Extensões

1. Adicionar subset de variáveis categóricas via One-Hot (ex: `JobRole`, `OverTime`).
2. Salvar scaler em `kmeans_scaler.joblib` para uso em produção.
3. Comparar inércia e Silhouette antes/depois de incluir categóricas.
4. Avaliar PCA (variância 90–95%) e medir impacto em separabilidade.
5. Introduzir detecção de outliers (IQR ou z-score) e reprocessar.

## Checklist

| Item | Status |
|------|--------|
| Seleção numéricas | OK |
| Remoção IDs/constantes | OK |
| Estratégia imputação definida | OK |
| Escalonamento aplicado | OK |
| Persistência dataset | OK |
| Plano de extensão categóricas | OK |
| Riscos mapeados | OK |

