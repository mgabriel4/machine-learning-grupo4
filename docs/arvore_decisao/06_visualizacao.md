# 06 - Visualização

Objetivo

- Fornecer visualizações legíveis da nossa árvore de decisão e da árvore completa, e exportar regras em texto.

Imagens

- `imagens/arvore_reduzida.png` — nossa árvore de decisão para apresentação.
- `imagens/decision_tree.png` — visualização da árvore completa (cortada a uma profundidade para visualização).

Trechos de código relevantes

```python
# Treinar nossa árvore de decisão (já salva como imagem)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
clf_small = DecisionTreeClassifier(max_depth=3, min_samples_split=20, random_state=42)
clf_small.fit(X_proc_sample, y.values)

# Salvar plot e regras
plot_tree(clf_small, feature_names=feature_names, class_names=['No','Yes'], filled=True, rounded=True, fontsize=10)
# salvar em imagens/arvore_reduzida.png
rules = export_text(clf_small, feature_names=list(feature_names))
with open('arvore_reduzida_rules.txt', 'w', encoding='utf-8') as f:
    f.write(rules)
```

Visualização incorporada

![Ávore de decisão reduzida](imagens/arvore_reduzida.png)
![Ávore de decisão completa](imagens/decision_tree.png)
