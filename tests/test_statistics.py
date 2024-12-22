import pytest
from src.datalib.statistics import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    perform_t_test,
    perform_chi_square_test,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
)
import numpy as np
import pandas as pd

def test_calculate_mean():
    """Test calculating the mean."""
    data = [1, 2, 3, 4, 5]
    assert calculate_mean(data) == 3

def test_calculate_median():
    """Test calculating the median."""
    data = [1, 3, 5, 2, 4]
    assert calculate_median(data) == 3

def test_calculate_mode():
    """Test calculating the mode."""
    data = [1, 1, 2, 3, 3, 3, 4]
    assert calculate_mode(data) == 3

def test_perform_t_test():
    """Test performing a t-test."""
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 3, 4, 5, 6]
    t_stat, p_value = perform_t_test(data1, data2)
    assert p_value > 0.05  # Assuming no significant difference

def test_perform_chi_square_test():
    """Test performing a chi-square test."""
    table = pd.DataFrame([[10, 20], [20, 40]])
    chi2, p, dof, expected = perform_chi_square_test(table)
    assert p > 0.05  # Assuming no significant association

def test_calculate_pearson_correlation():
    """Test calculating Pearson correlation."""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    corr, p_value = calculate_pearson_correlation(x, y)
    assert pytest.approx(corr) == 1

def test_calculate_spearman_correlation():
    """Test calculating Spearman correlation."""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    corr, p_value = calculate_spearman_correlation(x, y)
    assert pytest.approx(corr) == 1
