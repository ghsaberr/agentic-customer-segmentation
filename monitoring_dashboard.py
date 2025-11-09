# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px
import numpy as np
import os


st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

# --- CONFIG ---
DATA_PATH = "data/clustered_customers.parquet"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.title("📊 Customer Segmentation & KPI Monitoring Dashboard")

# --- LOAD CLUSTER DATA ---
if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
    st.sidebar.success("✅ Clustered dataset loaded.")
else:
    st.sidebar.error("❌ No clustered dataset found. Run /compute_clusters first.")
    st.stop()

# --- KPI SECTION ---
st.header("Key Cluster KPIs")

kpi_df = (
    df.groupby("cluster_kmeans")
    .agg(
        cluster_size=("customer_id", "count"),
        churn_rate=("churned", "mean"),
        avg_ltv=("monetary_ltv", "mean"),
        avg_engagement=("engagement_score", "mean")
    )
    .reset_index()
)

fig1 = px.bar(kpi_df, x="cluster_kmeans", y="cluster_size", title="Cluster Sizes", color="cluster_kmeans")
fig2 = px.bar(kpi_df, x="cluster_kmeans", y="churn_rate", title="Churn Rate per Cluster", color="cluster_kmeans")
fig3 = px.bar(kpi_df, x="cluster_kmeans", y="avg_ltv", title="Average LTV per Cluster", color="cluster_kmeans")
fig4 = px.bar(kpi_df, x="cluster_kmeans", y="avg_engagement", title="Avg Engagement", color="cluster_kmeans")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)
col3.plotly_chart(fig3, use_container_width=True)
col4.plotly_chart(fig4, use_container_width=True)

# --- MLflow EXPERIMENT MONITORING ---
st.header("📈 MLflow Experiment Tracking")

mlflow.set_tracking_uri(MLFLOW_URI)
try:
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.list_experiments()
    except AttributeError:
        experiments = mlflow.search_experiments()  # compatible with older versions

    exp_names = [e.name for e in experiments]
    exp_choice = st.selectbox("Select Experiment", exp_names)
    exp_id = [e.experiment_id for e in experiments if e.name == exp_choice][0]
    runs = mlflow.search_runs(experiment_ids=[exp_id])
    if not runs.empty:
        st.dataframe(runs[["run_id", "params.n_clusters", "metrics.silhouette", "start_time"]])
        st.plotly_chart(
            px.line(
                runs,
                x="start_time",
                y="metrics.silhouette",
                title="Silhouette Score Over Time"
            ),
            use_container_width=True
        )
    else:
        st.info("No runs logged yet.")
except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")

# --- DRIFT DETECTION (Simple Example) ---
st.header("⚠️ Drift Monitoring (Experimental)")

current_sizes = kpi_df["cluster_size"]
mean_size = current_sizes.mean()
std_size = current_sizes.std()

alerts = []
for i, row in kpi_df.iterrows():
    if abs(row["cluster_size"] - mean_size) > 2 * std_size:
        alerts.append(f"Cluster {row['cluster_kmeans']} size drift detected!")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("✅ No significant cluster drift detected.")
