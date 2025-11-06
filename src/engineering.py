import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# === Helper function for engagement proxy ===
def compute_engagement_response_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Compute engagement_response_proxy based on engagement_score, portal_logins,
    support_calls, and emails_opened.
    """
    df = df.copy()
    df['portal_per_call'] = df['portal_logins'] / (df['support_calls'] + 1)
    df['emails_per_portal'] = df['emails_opened'] / (df['portal_logins'] + 1)

    proxy = (
        (df['engagement_score'] * 0.6)
        + (np.clip(df['portal_per_call'], 0, 50) / 50 * 0.2)
        + (np.clip(df['emails_per_portal'], 0, 50) / 50 * 0.2)
    )
    return proxy


# === Main Feature Engineering Function ===
def run_feature_engineering(df, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    df_feat = df.copy()

    # ===== Feature Construction =====
    df_feat['recency_days'] = df_feat['days_since_last_claim'].fillna(
        (pd.to_datetime(df_feat['policy_end_date']) - pd.to_datetime(df_feat['policy_start_date'])).dt.days
    )
    df_feat['interactions_total'] = df_feat['portal_logins'] + df_feat['emails_opened'] + df_feat['support_calls']
    df_feat['interactions_per_12m'] = df_feat['interactions_total'] / (df_feat['tenure_months'] / 12 + 1e-6)
    df_feat['monetary_premium'] = df_feat['premium_amount']
    df_feat['monetary_ltv'] = df_feat['customer_ltv']

    # ✅ Use helper function here
    df_feat['engagement_response_proxy'] = compute_engagement_response_proxy(df_feat)

    df_feat['claim_severity_avg'] = np.where(
        df_feat['num_claims'] > 0, df_feat['claim_costs'] / df_feat['num_claims'], 0.0
    )
    df_feat['claims_per_12m'] = df_feat['num_claims'] / (df_feat['tenure_months'] / 12 + 1e-6)

    df_feat['logins_per_month'] = df_feat['portal_logins'] / (df_feat['tenure_months'] + 1e-6)
    for m in [3, 6, 12]:
        df_feat[f'logins_{m}m_proxy'] = df_feat['logins_per_month'] * m

    df_feat['claims_per_month'] = df_feat['num_claims'] / (df_feat['tenure_months'] + 1e-6)
    df_feat['claims_12m_proxy'] = df_feat['claims_per_month'] * 12

    region_median = df_feat.groupby('region')['claim_costs'].median().to_dict()
    df_feat['region_median_claim'] = df_feat['region'].map(region_median)
    df_feat['geo_risk_bucket'] = pd.qcut(
        df_feat['region_median_claim'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High']
    )

    df_feat['acquisition_cohort'] = pd.to_datetime(df_feat['policy_start_date']).dt.to_period('M').astype(str)
    bins = [0, 6, 12, 24, 48, 84, 10000]
    labels = ['0-6m', '6-12m', '12-24m', '24-48m', '48-84m', '84m+']
    df_feat['policy_age_bucket'] = pd.cut(df_feat['tenure_months'], bins=bins, labels=labels, right=False)

    df_feat['bundle_candidate_flag'] = (
        (df_feat['multi_policy_flag'] == 1) & (df_feat['customer_ltv'] > df_feat['customer_ltv'].median())
    ).astype(int)

    # ===== Scaling & Encoding =====
    numeric_features = [
        'age','income_level','premium_amount','tenure_months','portal_logins','emails_opened','support_calls',
        'payment_regularity','interactions_total','interactions_per_12m','monetary_premium','monetary_ltv',
        'engagement_score','engagement_response_proxy','claim_severity_avg','claims_per_12m','claims_12m_proxy',
        'logins_3m_proxy','logins_6m_proxy','logins_12m_proxy','delinquency_count','recency_days'
    ]
    numeric_features = [c for c in numeric_features if c in df_feat.columns]

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_feat[numeric_features].fillna(0))

    cat_to_encode = ['gender','region','policy_type','lifecycle_stage','geo_risk_bucket','policy_age_bucket']
    cat_to_encode = [c for c in cat_to_encode if c in df_feat.columns]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(df_feat[cat_to_encode].fillna('NA')) if len(cat_to_encode) > 0 else np.empty((len(df_feat), 0))

    # ===== Correlation Heatmap =====
    corr = pd.DataFrame(X_num, columns=numeric_features).corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect='auto')
    plt.xticks(range(len(numeric_features)), numeric_features, rotation=90, fontsize=8)
    plt.yticks(range(len(numeric_features)), numeric_features, fontsize=8)
    plt.colorbar()
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # ===== Feature Importance via RF (Proxy) =====
    y = df_feat['churned'].astype(int).values
    X_clust = np.hstack([X_num, X_cat])
    X_train, X_test, y_train, y_test = train_test_split(X_clust, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    ohe_feature_names = ohe.get_feature_names_out(cat_to_encode).tolist() if len(cat_to_encode) > 0 else []
    feature_names = numeric_features + ohe_feature_names
    fi_df = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)

    # Save plots & features
    fi_path = os.path.join(output_dir, "feature_importances.png")
    fi_df.head(20).plot.barh(x='feature', y='importance', title='Top Feature Importances', figsize=(8,6))
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()

    out_path = os.path.join(output_dir, "processed_features.parquet")
    to_save = df_feat[['customer_id'] + numeric_features + cat_to_encode + ['bundle_candidate_flag','multi_policy_flag','churned']]
    to_save.to_parquet(out_path, index=False)

    print(f"✅ Feature engineering completed. Processed shape: {X_clust.shape}")
    print(f"Saved files:\n- {heatmap_path}\n- {fi_path}\n- {out_path}")

    return df_feat, X_clust, fi_df
