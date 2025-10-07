from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def build_classifier_pipeline(preprocessor, classifier) -> Pipeline:
    """Monta pipeline padrão (preprocessador + estimador)."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', classifier)
    ])


def gridsearch_classifier(pipeline: Pipeline, X, y, param_grid: Dict[str, Any], cv_splits=5, scoring='f1', verbose=0) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=verbose)
    gs.fit(X, y)
    return gs


def build_decision_tree_pipeline(preprocessor, max_depth=None, random_state=42):
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    return build_classifier_pipeline(preprocessor, clf)


def build_knn_pipeline(preprocessor):
    return build_classifier_pipeline(preprocessor, KNeighborsClassifier())


def default_knn_param_grid():
    return {
        'clf__n_neighbors': [3,5,7,9],
        'clf__weights': ['uniform','distance'],
        'clf__p': [1,2]
    }


def default_tree_param_grid():
    return {
        'clf__criterion': ['gini','entropy'],
        'clf__max_depth': [None,5,10,15],
        'clf__min_samples_split': [2,5,10]
    }


def train_kmeans(X: pd.DataFrame, n_clusters=2, random_state=42) -> Tuple[KMeans, pd.Series]:
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
    labels = km.fit_predict(X)
    return km, pd.Series(labels, name='cluster')
