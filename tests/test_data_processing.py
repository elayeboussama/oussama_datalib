import pytest
from src.datalib.data_processing import load_csv, normalize_column, fill_missing_values, encode_categorical, scale_data
import pandas as pd
import numpy as np

def test_normalize_column(sample_dataframe):
    """Test normalizing a column."""
    df = normalize_column(sample_dataframe.copy(), "Value1")
    assert pytest.approx(df["Value1"].mean(), 0.1) == 0
    assert pytest.approx(df["Value1"].std(), 0.1) == 1

def test_fill_missing_values():
    """Test filling missing values."""
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    df_filled = fill_missing_values(df, "A", method="mean")
    assert not df_filled["A"].isna().any()

def test_encode_categorical(sample_dataframe):
    """Test encoding categorical values."""
    df = encode_categorical(sample_dataframe.copy(), "Category")
    assert "Category" in df.columns
    assert df["Category"].dtype == int
