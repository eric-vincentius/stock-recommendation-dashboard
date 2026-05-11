# pages/clustering.py
import streamlit as st
from utils.scoring import compute_features
import pandas as pd
from models.clustering import (
    prepare_clustering_data,
    scale_features,
    get_cluster_profile,
    interpret_cluster,
    run_kmeans,
    compute_elbow
)

def show():
    @st.cache_data(ttl=3600)
    def load_data():
        saham_data = pd.read_csv("data/saham_data.csv")
        latest = pd.read_csv("data/latest_snapshot.csv")
        summary = pd.read_csv("data/stock_summary.csv")
        return saham_data, latest, summary

    saham_data, latest, summary = load_data()

    # Prepare clustering
    df_cluster = prepare_clustering_data(summary)
    X_scaled, scaler = scale_features(df_cluster)
    df_cluster, kmeans = run_kmeans(df_cluster, X_scaled, k=3)

    profile = get_cluster_profile(df_cluster, kmeans, scaler)
    profile["Label"] = profile.apply(interpret_cluster, axis=1)

    df_cluster = df_cluster.merge(
        profile[["Cluster", "Label"]],
        on="Cluster",
        how="left"
    )

    st.subheader("📍 Cluster Visualization")

    st.scatter_chart(
        df_cluster.set_index("Stock_Name")[["Mean_Return", "Risk"]]
    )

    st.subheader("📊 Cluster Profiles")
    st.dataframe(profile)

    st.subheader("📈 Cluster Distribution")
    cluster_counts = df_cluster["Cluster"].value_counts()
    st.bar_chart(cluster_counts)

    st.subheader("Stock's Cluster Assignments")
    st.dataframe(df_cluster[["Stock_Name", "Mean_Return", "Risk", "Avg_Volume", "Label"]])