import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

root = r"c:\Users\ediad\OneDrive\Documentos\GitHub\machine-learning-grupo4\docs\knn"

y_test = pd.read_csv(root + "\\knn_y_test.csv")
y_pred = pd.read_csv(root + "\\knn_y_pred.csv")

if 'Attrition' in y_test.columns:
    y_true = y_test['Attrition']
else:
    y_true = y_test.iloc[:,0]
if 'y_pred' in y_pred.columns:
    yhat = y_pred['y_pred']
else:
    yhat = y_pred.iloc[:,0]

acc = accuracy_score(y_true, yhat)
prec = precision_score(y_true, yhat, zero_division=0)
rec = recall_score(y_true, yhat, zero_division=0)
f1 = f1_score(y_true, yhat, zero_division=0)
report = classification_report(y_true, yhat, zero_division=0)
cm = confusion_matrix(y_true, yhat)

print('ACC:', acc)
print('PREC:', prec)
print('REC:', rec)
print('F1:', f1)
print('\nCLASSIFICATION_REPORT:\n')
print(report)
print('\nCONFUSION_MATRIX:\n')
print(cm)
