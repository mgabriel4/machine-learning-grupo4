# 05 - Avaliação

Objetivo

- Avaliar o modelo no conjunto de teste usando métricas (Accuracy, Precision, Recall, F1), matriz de confusão e ROC AUC quando aplicável.


Gráficos e evidências de avaliação

- `imagens/confusion_matrix.png` — matriz de confusão (visualiza TP/TN/FP/FN).
- `imagens/roc_curve.png` — curva ROC (se `predict_proba` estiver disponível).

Relatório de classificação (valores extraídos do teste)

| label | precision | recall | f1-score | support |
|-------|----------:|-------:|---------:|-------:|
| 0 (No) | 0.8735 | 0.8947 | 0.8840 | 247 |
| 1 (Yes) | 0.3659 | 0.3191 | 0.3409 | 47 |
| accuracy |  |  | 0.8027 | 294 |
| macro avg | 0.6197 | 0.6069 | 0.6125 | 294 |
| weighted avg | 0.7924 | 0.8027 | 0.7972 | 294 |

Observação: os valores acima são os números exatos obtidos no conjunto de teste (294 amostras). Eles resumem a capacidade do modelo em distinguir entre funcionários que saem (1) e que permanecem (0).

Trechos de código relevantes

```python
# Prever e calcular métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print('F1:', f1)

# Matriz de confusão e salvar imagem
cm = confusion_matrix(y_test, y_pred)
# (plot e salvamento)

# Salvar classification report
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('classification_report_test_arvore.csv')
```


![Matriz de confusão](imagens/confusion_matrix.png)
![Curva ROC](imagens/roc_curve.png)



Interpretação e explicações (com exemplos embutidos)

1) Por que a performance é assimétrica?

- A classe majoritária (`No`) tem 1.235 exemplos no dataset completo, enquanto a classe minoritária (`Yes`) tem 235 — com essa diferença o classificador aprende melhor os padrões da classe majoritária, levando a recall/precision mais altas para `No`.

2) Exemplos de falsos negativos (casos reais `Yes` classificados como `No`) — trechos representativos do conjunto de teste (colunas-chave):

```csv
Age,JobRole,OverTime,MonthlyIncome,TotalWorkingYears,y_true,y_pred
31,Manufacturing Director,Yes,6179,10,1,0
41,Research Director,No,19545,23,1,0
31,Sales Executive,Yes,4559,4,1,0
58,Healthcare Representative,No,10312,40,1,0
44,Human Resources,No,10482,24,1,0
```

Justificativa: nos exemplos acima, fatores como altos níveis de remuneração ou longa experiência (por exemplo, 19545 de MonthlyIncome e 23 anos de TotalWorkingYears) podem induzir o modelo a classificar como estável (`No`), mesmo quando o rótulo real era `Yes` — possivelmente porque esses perfis aparecem frequentemente na base como `No`.

3) Exemplos de falsos positivos (casos reais `No` classificados como `Yes`) — trechos representativos do conjunto de teste:

```csv
Age,JobRole,OverTime,MonthlyIncome,TotalWorkingYears,y_true,y_pred
18,Research Scientist,No,1514,0,0,1
51,Laboratory Technician,No,2838,8,0,1
27,Research Scientist,Yes,2478,1,0,1
23,Laboratory Technician,No,3295,1,0,1
33,Laboratory Technician,No,2028,14,0,1
```

Justificativa: esses perfis têm atributos (baixa remuneração, pouca experiência, trabalhos com alta rotatividade em histórico) que o modelo associa a risco, resultando em previsões positivas mesmo quando o rótulo era `No`.

Análise de erros

- As amostras exemplificadas acima (trechos CSV) mostram os padrões mais frequentes de erro: falsos negativos frequentemente têm altos valores de `MonthlyIncome` e `TotalWorkingYears`, enquanto falsos positivos aparecem com `MonthlyIncome` e `TotalWorkingYears` baixos ou perfis de início de carreira.

- Para reduzir esses erros recomenda-se investigar interações entre `OverTime` x `MonthlyIncome`, introduzir variáveis derivadas (ex.: `income_per_year_of_experience`) e testar reponderação/oversampling.