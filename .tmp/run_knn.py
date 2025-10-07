import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ROOT = r"c:\Users\ediad\OneDrive\Documentos\GitHub\machine-learning-grupo4\docs\knn"
CSV = os.path.join(os.getcwd(), 'docs', 'knn', 'funcionarios.csv')
if not os.path.exists(CSV):
    print('CSV not found at', CSV)
    sys.exit(1)

print('Loading', CSV)
df = pd.read_csv(CSV)
print('Shape:', df.shape)

target = 'Attrition'
if df[target].dtype == 'object':
    y = df[target].map({'Yes':1, 'No':0})
else:
    y = df[target]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in ['EmployeeNumber','EmployeeCount','StandardHours']:
    if c in num_cols: num_cols.remove(c)
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if target in cat_cols: cat_cols.remove(target)

print('Num cols:', len(num_cols), 'Cat cols:', len(cat_cols))

X = df[num_cols + cat_cols].copy()
# Impute numeric by median (fallback) and ensure no NA
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print('Train/Test shapes:', X_train.shape, X_test.shape)

num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)], remainder='drop')

pipe = Pipeline([('preprocessor', preprocessor), ('clf', KNeighborsClassifier())])
param_grid = {'clf__n_neighbors':[3,5,7,9], 'clf__weights':['uniform','distance'], 'clf__p':[1,2]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Starting GridSearch...')
start = time.time()
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
try:
    gs.fit(X_train, y_train)
except Exception as e:
    print('GridSearch failed:', e)
    sys.exit(1)
end = time.time()
print('GridSearch completed in', round(end-start,2), 's')
print('Best params:', gs.best_params_)
print('Best CV f1:', gs.best_score_)

best_model = gs.best_estimator_

y_pred = best_model.predict(X_test)
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
img_dir = os.path.join(ROOT, 'imagens')
os.makedirs(img_dir, exist_ok=True)
plt.savefig(os.path.join(img_dir, 'knn_confusion_matrix.png'), bbox_inches='tight', dpi=150)
plt.close()

os.makedirs(ROOT, exist_ok=True)
joblib.dump(best_model, os.path.join(ROOT, 'knn_model.joblib'))
pd.DataFrame(y_test.reset_index(drop=True)).to_csv(os.path.join(ROOT, 'knn_y_test.csv'), index=False)
pd.DataFrame(y_pred, columns=['y_pred']).to_csv(os.path.join(ROOT, 'knn_y_pred.csv'), index=False)

print('Saved model and predictions to', ROOT)
