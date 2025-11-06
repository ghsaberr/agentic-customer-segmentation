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

def run_clustering_pipeline(input_path="data/processed_features.parquet", output_path="data/clustered_customers.parquet"):
    df_feat = pd.read_parquet(input_path)

    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['bundle_candidate_flag','multi_policy_flag','churned']]
    X = df_feat[num_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate cluster counts
    inertia, silhouette_scores = [], []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(K_range, inertia, marker='o')
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("k"); plt.ylabel("Inertia")

    plt.subplot(1,2,2)
    plt.plot(K_range, silhouette_scores, marker='o', color='orange')
    plt.title("Silhouette Scores")
    plt.xlabel("k"); plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    best_k = 5
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_feat['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    df_feat['pca1'], df_feat['pca2'] = pca.fit_transform(X_scaled).T

    reducer = umap.UMAP(random_state=42)
    df_feat['umap1'], df_feat['umap2'] = reducer.fit_transform(X_scaled).T

    plt.figure(figsize=(7,6))
    sns.scatterplot(x='umap1', y='umap2', hue='cluster_kmeans', data=df_feat, palette='tab10', s=20)
    plt.title("Customer Segments (UMAP projection)")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    gmm = GaussianMixture(n_components=best_k, random_state=42)
    df_feat['cluster_gmm'] = gmm.fit_predict(X_scaled)

    print("KMeans silhouette:", silhouette_score(X_scaled, df_feat['cluster_kmeans']))
    print("KMeans Davies-Bouldin:", davies_bouldin_score(X_scaled, df_feat['cluster_kmeans']))
    print("GMM silhouette:", silhouette_score(X_scaled, df_feat['cluster_gmm']))

    cluster_profile = df_feat.groupby('cluster_kmeans')[num_cols + ['churned']].agg(['mean','median'])
    print("\nCluster Profile Summary:")
    display(cluster_profile.head())

    df_feat.to_parquet(output_path, index=False)
    print(f"✅ Clustered dataset saved at: {output_path}")

    return df_feat, cluster_profile
