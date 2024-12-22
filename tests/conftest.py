import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification

@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame."""
    data = {
        "Category": ["A", "B", "C", "A", "B", "C"],
        "Value1": [10, 20, 15, 30, 25, 35],
        "Value2": [1.5, 2.0, 1.8, 2.5, 2.3, 3.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def regression_data():
    """Fixture to provide sample data for regression."""
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    return X, y

@pytest.fixture
def classification_data():
    """Fixture to provide sample data for classification."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    return X, y
