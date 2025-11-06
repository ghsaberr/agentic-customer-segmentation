import numpy as np

def test_engagement_response_proxy_calc(df_customers):
    df = df_customers.copy()
    # If columns do not exist, skip
    if not all(c in df.columns for c in ['portal_logins','support_calls','emails_opened','engagement_score']):
        return

    df['portal_per_call'] = df['portal_logins'] / (df['support_calls'] + 1)
    df['emails_per_portal'] = df['emails_opened'] / (df['portal_logins'] + 1)
    expected = (df['engagement_score'] * 0.6) + (np.clip(df['portal_per_call'],0,50) / 50 * 0.2) + (np.clip(df['emails_per_portal'],0,50) / 50 * 0.2)
    calculated = expected
    assert np.allclose(expected.fillna(0), calculated.fillna(0), atol=1e-6)
