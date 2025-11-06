import pandas as pd
import re

def clean_customer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure date columns are datetimes
    for col in ['policy_start_date', 'policy_end_date', 'last_claim_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fill or flag missing numeric fields
    # Days-since-last-claim: keep NaN for no-claims, but add a boolean flag
    df['has_claims'] = (df['num_claims'] > 0).astype(int)
    df['days_since_last_claim_flag'] = df['days_since_last_claim'].isna().astype(int)

    # Impute missing engagement_score with median and add flag
    if 'engagement_score' in df.columns:
        df['engagement_score_missing'] = df['engagement_score'].isna().astype(int)
        df['engagement_score'] = df['engagement_score'].fillna(df['engagement_score'].median())

    # Numeric clipping for unreasonable values
    df['income_level'] = df['income_level'].clip(lower=10_000, upper=500_000)
    df['premium_amount'] = df['premium_amount'].clip(lower=50, upper=100000)

    # Convert categorical fields to categories (memory + explicit dtype)
    cat_cols = ['gender', 'region', 'policy_type', 'payment_frequency', 'lifecycle_stage', 'engagement_trend']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Feature: claims per year
    df['claims_per_year'] = df['num_claims'] / (df['tenure_months'] / 12 + 1e-6)

    # Keep an audit column listing applied cleaning steps (simple)
    df['__cleaned_on'] = pd.Timestamp.now()

    return df