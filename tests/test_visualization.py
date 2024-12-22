import pytest
from src.datalib.visualization import plot_bar, plot_histogram, plot_correlation_matrix, plot_scatter
import pandas as pd

def test_plot_bar(sample_dataframe, mocker):
    """Test the bar plot function."""
    mock_show = mocker.patch("matplotlib.pyplot.show")
    plot_bar(sample_dataframe, x="Category", y="Value1", title="Test Bar Plot")
    mock_show.assert_called_once()

def test_plot_histogram(sample_dataframe, mocker):
    """Test the histogram plot function."""
    mock_show = mocker.patch("matplotlib.pyplot.show")
    plot_histogram(sample_dataframe, column="Value1", bins=5, title="Test Histogram")
    mock_show.assert_called_once()

def test_plot_correlation_matrix(sample_dataframe, mocker):
    """Test the correlation matrix plot function."""
    mock_show = mocker.patch("matplotlib.pyplot.show")
    numeric_data = sample_dataframe[["Value1", "Value2"]]
    plot_correlation_matrix(numeric_data, title="Test Correlation Matrix")
    mock_show.assert_called_once()

def test_plot_scatter(sample_dataframe, mocker):
    """Test the scatter plot function."""
    mock_show = mocker.patch("matplotlib.pyplot.show")
    plot_scatter(sample_dataframe, x="Value1", y="Value2", title="Test Scatter Plot")
    mock_show.assert_called_once()
