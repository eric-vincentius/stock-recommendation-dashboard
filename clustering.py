# clustering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# CLUSTER SUMMARY
# =========================
def get_cluster_summary(df):
    summary = df.groupby("Cluster").agg({
        "Mean_Return": "mean",
        "Risk": "mean",
        "Avg_Volume": "mean",
        "Stock_Name": "count"
    }).rename(columns={"Stock_Name": "Num_Stocks"}).reset_index()

    return summary

# =========================
# STOCK LIST PER CLUSTER
# =========================
def get_cluster_members(df):
    clusters = {}

    for cluster_id in sorted(df["Cluster"].unique()):
        clusters[cluster_id] = df[df["Cluster"] == cluster_id]["Stock_Name"].tolist()

    return clusters