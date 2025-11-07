import pandas as pd
import numpy as np
from pathlib import Path
from src.cluster_profiler import cluster_profiler

def test_cluster_profiler_counts(df_customers):
    df = df_customers.copy()
    labels = [i % 2 for i in range(len(df))]
    df['cluster_kmeans'] = labels
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    assert cluster_profiler is not None, "cluster_profiler not found (import path differs)"
    prof = cluster_profiler(0, df, numeric_cols=num_cols)
    assert prof['cluster_id'] == 0
    assert 'count' in prof and prof['count'] >= 1

