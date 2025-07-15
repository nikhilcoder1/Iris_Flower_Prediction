import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
