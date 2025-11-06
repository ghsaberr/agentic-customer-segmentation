import pytest
import pandas as pd
import os

# If the processed file exists, otherwise create a small dataframe.
@pytest.fixture(scope="session")
def df_customers():
    path = os.path.join("data", "processed_features.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df.head(200)  # A small sample is sufficient for tests
    # fallback: Creating a sample dataframe
    df = pd.DataFrame({
        "customer_id": ["C1","C2","C3","C4"],
        "age": [30,45,60,25],
        "income_level": [40000, 80000, 30000, 25000],
        "premium_amount": [800,1200,600,450],
        "tenure_months": [12,48,6,36],
        "portal_logins": [10,2,0,5],
        "emails_opened": [8,1,0,3],
        "support_calls": [0,3,1,2],
        "num_claims": [0,1,0,2],
        "claim_costs": [0.0,1500.0,0.0,3000.0],
        "policy_start_date": pd.to_datetime(["2022-01-01","2020-06-01","2024-01-01","2021-07-01"]),
        "policy_end_date": pd.to_datetime(["2024-12-31","2024-12-31","2024-12-31","2024-12-31"]),
        "engagement_score": [0.7,0.3,0.1,0.45],
        "churned": [0,1,0,0],
        "multi_policy_flag": [1,0,0,1],
        "customer_ltv": [2000,4000,1200,1800],
    })
    return df
