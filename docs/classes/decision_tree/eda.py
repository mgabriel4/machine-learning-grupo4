import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/mgabriel4/Documentos/GitHub/machine-learning-grupo4/data/funcionarios.csv')

print('\nPrimeiras 5 linhas do dataset:')
print(df.head())

print('\nInformações do dataset:')
print(df.info())

print('\nEstatísticas descritivas do dataset:')
print(df.describe())

print('\nValores nulos por coluna:')
print(df.isnull().sum())

# Visualizações
plt.figure(figsize=(8, 6))
df['BusinessTravel'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), textprops={'fontsize': 12})
plt.title('Viagens de Negócios', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/business_travel.png', dpi=150)
plt.show()

plt.figure(figsize=(6, 6))
df['Attrition'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), textprops={'fontsize': 12})
plt.title('Funcionário saiu da empresa?', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/attrition_pie.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Department', data=df, palette='Set3', order=df['Department'].value_counts().index)
plt.title('Departamentos', fontsize=16, fontweight='bold')
plt.xlabel('Departamento', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/department.png', dpi=150)
plt.show()

plt.figure(figsize=(6, 5))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), textprops={'fontsize': 12})
plt.title('Gênero', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/gender.png', dpi=150)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='JobRole', data=df, palette='Spectral', order=df['JobRole'].value_counts().index)
plt.title('Funções', fontsize=16, fontweight='bold')
plt.xlabel('Função', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/jobrole.png', dpi=150)
plt.show()

plt.figure(figsize=(6, 5))
df['MaritalStatus'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), textprops={'fontsize': 12})
plt.title('Estado Civil', fontsize=16, fontweight='bold')
plt.xlabel('Estado Civil', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/marital_status.png', dpi=150)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='OverTime', data=df, palette='Spectral', order=df['OverTime'].value_counts().index)
plt.title('Horas Extras', fontsize=16, fontweight='bold')
plt.xlabel('Horas Extras', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('./docs/classes/decision_tree/img/overtime.png', dpi=150)
plt.show()
