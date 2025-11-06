import numpy as np

def cluster_profiler(cluster_id, df_clustered, numeric_cols):
    members = df_clustered[df_clustered['cluster_kmeans'] == cluster_id]
    n = len(members)
    profile = {
        "cluster_id": int(cluster_id),
        "count": int(n),
        "churn_rate": float(members['churned'].mean()) if n>0 else None,
        "median_ltv": float(members['monetary_ltv'].median()) if n>0 else None,
        "avg_engagement": float(members['engagement_score'].mean()) if n>0 else None,
        "avg_claims_per_year": float(members['claims_per_12m'].mean()) if n>0 else None,
        "top_numeric_features": {}
    }
    if n>0:
        global_meds = df_clustered[numeric_cols].median()
        diffs = (members[numeric_cols].median() - global_meds).abs().sort_values(ascending=False)
        profile["top_numeric_features"] = diffs.to_dict()
    return profile
