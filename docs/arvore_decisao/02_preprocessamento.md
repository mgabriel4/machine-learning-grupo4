# 02 - Pré-processamento

Objetivo

- Preparar os dados para o modelo: imputação, encoding de categóricas, remoção de colunas irrelevantes e escalonamento das numéricas.

Arquivo processado

- `arvore_decisao_processed.csv` — dataset resultante após aplicar imputação, encoding e escalonamento (salvo pelo notebook).

Resumo do CSV processado

- Número de linhas (registros): 1470 (dataset exemplo utilizado no notebook).

- Número de features após encoding: 51 (confirmado no arquivo `arvore_decisao_processed.csv`).

Esquema de colunas (head)

As primeiras colunas correspondem às features numéricas escalonadas (valores z-score no exemplo):

- Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

Em seguida, aparecem colunas resultantes do OneHotEncoder (exemplos):

- BusinessTravel_Non-Travel, BusinessTravel_Travel_Frequently, BusinessTravel_Travel_Rarely,

- Department_Human Resources, Department_Research & Development, Department_Sales,

- EducationField_Human Resources, EducationField_Life Sciences, EducationField_Marketing, EducationField_Medical, EducationField_Other, EducationField_Technical Degree,

- Gender_Female, Gender_Male,

- JobRole_Healthcare Representative, JobRole_Human Resources, JobRole_Laboratory Technician, JobRole_Manager, JobRole_Manufacturing Director, JobRole_Research Director, JobRole_Research Scientist, JobRole_Sales Executive, JobRole_Sales Representative,

- MaritalStatus_Divorced, MaritalStatus_Married, MaritalStatus_Single,

- OverTime_No, OverTime_Yes,

- Attrition_binary

Exemplo (primeira linha do CSV processado)

Os valores abaixo são a primeira linha do `arvore_decisao_processed.csv` — note que os numéricos já foram escalonados (z-score) e as categorias foram convertidas para dummies (0/1):

- Age: 0.4463504035345031

- DailyRate: 0.7425265337769018

- DistanceFromHome: -1.0109093429124179

- Education: -0.8916882501868245

- EnvironmentSatisfaction: -0.6605306743650393

- HourlyRate: 1.3831382668932737

- JobInvolvement: 0.3796721288811475

- JobLevel: -0.05778754527941421

- JobSatisfaction: 1.1532535902386967

- MonthlyIncome: -0.10834951351067117

- ... (followed by one-hot columns, e.g. Department_Research & Development = 1)

Observações práticas

- O pipeline usa SimpleImputer(strategy='median') para numéricos e SimpleImputer(strategy='most_frequent') para categóricas antes do OneHotEncoder(handle_unknown='ignore').

- O StandardScaler é aplicado às colunas numéricas após imputação; por isso o CSV contém valores escalonados.

- O `Attrition` original foi mapeado para `Attrition_binary` (Yes=1, No=0) e preservado como última coluna do CSV processado para facilitar treinos e análises.

Como reproduzir rapidamente (exemplo de pandas)

```python
import pandas as pd

# carregar
df = pd.read_csv('docs/arvore_decisao/arvore_decisao_processed.csv')

print(df.shape)  # (1470, 51)
print(df.columns.tolist())
print(df.iloc[0].to_dict())  # primeira linha — valores transformados
```

Trechos de código relevantes

```python
# Mapear alvo e remover colunas
df['Attrition_binary'] = df['Attrition'].map({'Yes':1,'No':0})
drop_cols = ['EmployeeCount','EmployeeNumber','Over18','StandardHours','Attrition']
df_proc = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Identificar colunas
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Aplicar e salvar
X_proc = preprocessor.fit_transform(X)
# Salvar como CSV (transformado para DataFrame com nomes de features)
```

Resultados

- Número de features após encoding: ver `arvore_decisao_processed.csv` (por ex. 51 features no dataset de exemplo).

- Pipeline salvo como parte do `arvore_decisao_pipeline.joblib` após o GridSearch (se executado).

Observações

- O notebook tenta preservar nomes de features usando `get_feature_names_out` do OneHotEncoder; se não estiver disponível na versão da sua biblioteca, ajustamos a chamada automaticamente.

---

## Visão geral das variáveis originais

| Tipo | Colunas (originais) | Qtde | Observações |
|------|----------------------|------|-------------|
| Numéricas (contínuas/ordinais) | Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager | 23 | Escalonadas via StandardScaler (robustez em comparação entre modelos). |
| Categóricas (nominais) | BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime, Attrition | 8 | One-Hot Encoding (handle_unknown=ignore). |
| Removidas | EmployeeCount, EmployeeNumber, Over18, StandardHours, Attrition (substituída) | 5 | Constantes ou substituídas por mapeamento. |
| Total consideradas antes do One-Hot | 23 + 8 = 31 | 31 | Base para derivar 51 após expansão dummies. |

### Cardinalidade das categóricas

| Coluna | Cardinalidade | Categorias (amostra) | Justificativa One-Hot |
|--------|---------------|----------------------|-----------------------|
| BusinessTravel | 3 | Non-Travel, Travel_Rarely, Travel_Frequently | Baixa cardinalidade; dummies claras. |
| Department | 3 | Sales, Research & Development, Human Resources | Poucas categorias; impacto direto em attrition. |
| EducationField | 6 | Life Sciences, Medical, Marketing, Technical Degree, Other, Human Resources | Não ordinal; preservar separação. |
| Gender | 2 | Male, Female | Binária; codificada para consistência. |
| JobRole | 9 | (diversos) | Maior granularidade; decisiva em splits da árvore. |
| MaritalStatus | 3 | Single, Married, Divorced | Possível relação com retenção/carga de trabalho. |
| OverTime | 2 | Yes, No | Forte sinal potencial de exaustão/saída. |
| Attrition | 2 | Yes, No | Alvo -> mapeado para Attrition_binary. |

> Não há necessidade de Target Encoding aqui porque a árvore lida bem com expansão de dummies e cardinalidades moderadas.

### Fluxo do Pré-processamento (Diagrama)

```mermaid
flowchart LR
    A[Dados brutos funcionarios.csv] --> B[Mapeia Attrition -> Attrition_binary]
    B --> C[Remove colunas irrelevantes]
    C --> D[Separa numéricas / categóricas]
    D --> E1[Imputer median (num)]
    D --> E2[Imputer most_frequent (cat)]
    E1 --> F1[StandardScaler]
    E2 --> F2[OneHotEncoder ignore]
    F1 --> G[Concatenação]
    F2 --> G
    G --> H[Dataset transformado (51 features)]
    H --> I[Modelagem (Decision Tree + GridSearch)]
```

### Gráfico de apoio sugerido

Um gráfico útil neste estágio é a distribuição de cardinalidade (número de categorias) ou de valores ausentes. Este dataset (IBM HR) tipicamente não possui valores nulos, então o foco informativo recai sobre cardinalidade.

Pseudo-gráfico (cardinalidade categóricas):

```
JobRole          ██████████ 9
EducationField   ██████     6
BusinessTravel   ███        3
Department       ███        3
MaritalStatus    ███        3
OverTime         ██         2
Gender           ██         2
Attrition        ██         2
```

Para gerar um gráfico real (salvar em `docs/arvore_decisao/imagens/cardinalidade_categoricas.png`):

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('docs/arvore_decisao/funcionarios.csv')
cat_cols = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime','Attrition']
card = df[cat_cols].nunique().sort_values(ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x=card.values, y=card.index, palette='viridis')
plt.title('Cardinalidade das Categóricas')
plt.xlabel('Número de categorias')
plt.tight_layout()
plt.savefig('docs/arvore_decisao/imagens/cardinalidade_categoricas.png', dpi=140)
```

Depois, inclua no Markdown onde desejar:

```markdown
![Cardinalidade das categóricas](imagens/cardinalidade_categoricas.png)
```

### Justificativas das principais decisões

| Decisão | Alternativas analisadas | Racional escolhido |
|---------|-------------------------|--------------------|
| Mediana para numéricos | Média, KNN Imputer | Mediana robusta a outliers; simples e rápida. |
| Moda para categóricas | Categoria fixa "Missing" | Dataset sem nulos reais; moda mantém distribuição. |
| OneHotEncoder(ignore) | OrdinalEncoder, Target Encoding | Evita supor ordens; `ignore` garante robustez a categorias inéditas. |
| StandardScaler | MinMaxScaler, sem escala | Facilita comparação de experimentos com modelos sensíveis a escala. |
| Remoção de EmployeeNumber/Count etc. | Manter tudo | Colunas constantes ou IDs não informativos para attrition. |
| Mapeamento Attrition -> binário | LabelEncoder | Mapeamento explícito torna leitura mais clara (Yes=1, No=0). |
