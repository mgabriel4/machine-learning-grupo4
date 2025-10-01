import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('/home/mgabriel4/Documentos/GitHub/machine-learning-grupo4/data/funcionarios.csv')

#tratamento de variáveis categóricas
#label encoding para variáveis binárias
df['BusinessTravel'] = LabelEncoder().fit_transform(df['BusinessTravel'])
df['Department'] = LabelEncoder().fit_transform(df['Department'])
df['EducationField'] = LabelEncoder().fit_transform(df['EducationField'])
df['JobRole'] = LabelEncoder().fit_transform(df['JobRole'])
df['MaritalStatus'] = LabelEncoder().fit_transform(df['MaritalStatus'])

#one-hot encoding para variáveis com mais de duas categorias
df = pd.get_dummies(df,columns=['Attrition','Over18','OverTime', 'Gender'], drop_first=True)

print(df.head())