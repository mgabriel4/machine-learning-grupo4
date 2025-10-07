# 01 - Exploração dos Dados

Objetivo

- Entender a estrutura dos dados, tipos, distribuições e possíveis problemas (valores faltantes, outliers, classes desbalanceadas).

Gráficos gerados

- `imagens/attrition_distribution.png` — distribuição do alvo `Attrition` com percentuais.
- `imagens/histogramas_numericas.png` — histogramas das variáveis numéricas.
- `imagens/boxplots_numericas.png` — boxplots para `Age`, `MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`.
- `imagens/correlation_heatmap.png` — heatmap da correlação entre variáveis numéricas.
- `imagens/count_jobrole.png`, `imagens/count_department.png`, `imagens/count_maritalstatus.png`, `imagens/count_educationfield.png` — contagens por categoria.
- `imagens/attrition_rate_by_jobrole.png` — taxa de attrition por `JobRole`.

Trechos de código relevantes

```python
# Carregar dados e visualizar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('funcionarios.csv')
print(df.shape)
print(df['Attrition'].value_counts())

# Plot distribuição do alvo
ax = sns.countplot(x='Attrition', data=df, palette='Set2')
# (código para salvar imagem omitido)
```

Resultados resumidos

- Observamos N linhas e M colunas (ver `arvore_decisao.ipynb` para números exatos).
- Classe `Attrition` apresenta desbalanceamento típico com maioria `No`.
- Algumas variáveis mostram skew alto (detalhes no notebook) e `MonthlyIncome`/`OverTime` aparecem entre variáveis relevantes nas análises iniciais.

![Distribuição do alvo](imagens/attrition_distribution.png)
![Histogramas das variáveis numéricas](imagens/histogramas_numericas.png)
![Boxplots](imagens/boxplots_numericas.png)
![Heatmap da correlação entre variáveis numéricas](imagens/correlation_heatmap.png)
![Contagens por categoria](imagens/count_jobrole.png)
![Contagens por categoria](imagens/count_department.png)
![Contagens por categoria]( imagens/count_maritalstatus.png)
![Contagens por categoria](imagens/count_educationfield.png)
![Taxa de attrition](imagens/attrition_rate_by_jobrole.png)

