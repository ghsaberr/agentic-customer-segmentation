import pandas as pd
import pytest

def test_no_negative_premiums(df_customers):
    assert (df_customers['premium_amount'] >= 0).all(), "Negative premium detected"

def test_valid_date_ranges(df_customers):
    # policy_start_date before policy_end_date
    if 'policy_start_date' in df_customers.columns and 'policy_end_date' in df_customers.columns:
        assert (pd.to_datetime(df_customers['policy_start_date']) <= pd.to_datetime(df_customers['policy_end_date'])).all()

def test_missingness_threshold(df_customers):
    miss = df_customers.isna().mean()
    # Allow up to 20% missing for columns (configurable)
    assert (miss < 0.2).all(), f"Too many missing values: {miss.to_dict()}"
