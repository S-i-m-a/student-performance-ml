from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

def build_pipeline(max_depth=4, seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(criterion="gini", max_depth=max_depth,
                                        min_samples_leaf=10, random_state=seed)),
    ])
