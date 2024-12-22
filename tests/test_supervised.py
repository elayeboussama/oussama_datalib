import pytest
from src.datalib.supervised import train_knn, train_decision_tree, train_random_forest

def test_train_knn(classification_data):
    """Test training a KNN classifier."""
    X, y = classification_data
    model = train_knn(X, y, n_neighbors=3)
    assert model.n_neighbors == 3

def test_train_decision_tree(classification_data):
    """Test training a decision tree classifier."""
    X, y = classification_data
    model = train_decision_tree(X, y, max_depth=5)
    assert model.max_depth == 5

def test_train_random_forest(classification_data):
    """Test training a random forest classifier."""
    X, y = classification_data
    model = train_random_forest(X, y, n_estimators=50)
    assert model.n_estimators == 50
