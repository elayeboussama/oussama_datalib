import pytest
from src.datalib.analysis import linear_regression, polynomial_regression, multiple_linear_regression

def test_linear_regression(regression_data):
    """Test linear regression model training."""
    X, y = regression_data
    model, metrics = linear_regression(X, y, return_metrics=True)
    assert metrics["R-squared"] > 0.9

def test_polynomial_regression(regression_data):
    """Test polynomial regression model training."""
    X, y = regression_data
    model, metrics = polynomial_regression(X, y, degree=2, return_metrics=True)
    assert metrics["R-squared"] > 0.9

def test_multiple_linear_regression(regression_data):
    """Test multiple linear regression model training."""
    X, y = regression_data
    model, metrics = multiple_linear_regression(X, y, return_metrics=True)
    assert metrics["R-squared"] > 0.9
