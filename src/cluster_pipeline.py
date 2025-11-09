import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import mlflow
from mlflow import log_param, log_metric, log_artifact
from IPython.display import display


def run_clustering_pipeline(
    input_path="data/processed_features.parquet",
    output_path="data/clustered_customers.parquet"
):
    # --- Load data ---
    df_feat = pd.read_parquet(input_path)
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['bundle_candidate_flag', 'multi_policy_flag', 'churned']]
    X = df_feat[num_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Evaluate cluster counts ---
    inertia, silhouette_scores = [], []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # --- Determine best k (simplified) ---
    best_k = 5
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_feat['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    # --- Dimensionality reduction ---
    pca = PCA(n_components=2, random_state=42)
    df_feat['pca1'], df_feat['pca2'] = pca.fit_transform(X_scaled).T

    reducer = umap.UMAP(random_state=42)
    df_feat['umap1'], df_feat['umap2'] = reducer.fit_transform(X_scaled).T

    # --- Compute metrics ---
    sil_kmeans = silhouette_score(X_scaled, df_feat['cluster_kmeans'])
    db_kmeans = davies_bouldin_score(X_scaled, df_feat['cluster_kmeans'])
    gmm = GaussianMixture(n_components=best_k, random_state=42)
    df_feat['cluster_gmm'] = gmm.fit_predict(X_scaled)
    sil_gmm = silhouette_score(X_scaled, df_feat['cluster_gmm'])

    cluster_profile = df_feat.groupby('cluster_kmeans')[num_cols + ['churned']].agg(['mean', 'median'])
    df_feat.to_parquet(output_path, index=False)

    print("✅ Clustered dataset saved:", output_path)
    print("Silhouette (KMeans):", sil_kmeans)
    print("Davies-Bouldin (KMeans):", db_kmeans)
    print("Silhouette (GMM):", sil_gmm)

    # --- Log to MLflow ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer_segmentation"))

    with mlflow.start_run(run_name=f"kmeans_k{best_k}"):
        mlflow.log_param("n_clusters", best_k)
        mlflow.log_param("features_used", num_cols)
        mlflow.log_metric("silhouette_kmeans", sil_kmeans)
        mlflow.log_metric("davies_bouldin", db_kmeans)
        mlflow.log_metric("silhouette_gmm", sil_gmm)
        mlflow.log_artifact(output_path)
        cluster_summary_path = "data/cluster_profile_summary.csv"
        cluster_profile.to_csv(cluster_summary_path)
        mlflow.log_artifact(cluster_summary_path)

    return df_feat, cluster_profile
