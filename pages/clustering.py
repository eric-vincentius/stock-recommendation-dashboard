import streamlit as st
import pandas as pd
import plotly.express as px
from models.clustering import (
    prepare_clustering_data,
    scale_features,
    get_cluster_profile,
    interpret_cluster,
    run_kmeans
)

# =========================
# PAGE
# =========================
def show():

    # =========================
    # CLEAN MODERN CSS
    # =========================
    st.markdown("""
    <style>

    .block-container {
        padding-top: 1.5rem;
    }

    /* TITLE */
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 10px;
    }

    /* SECTION */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 10px;
    }

    /* CARD */
    .card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* TABLE */
    table {
        width: 100%;
        border-collapse: collapse;
    }

    thead {
        background: #f3f4f6;
    }

    th, td {
        padding: 10px;
        text-align: center;
    }

    tbody tr:hover {
        background: #f9fafb;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown(
        '<div class="main-title">Stock Clustering Analysis</div>',
        unsafe_allow_html=True
    )

    st.caption("Segmentasi saham berdasarkan return, risk, dan volume")

    # =========================
    # LOAD DATA
    # =========================
    @st.cache_data(ttl=3600)
    def load_data():
        saham_data = pd.read_csv("data/saham_data.csv")
        latest = pd.read_csv("data/latest_snapshot.csv")
        summary = pd.read_csv("data/stock_summary.csv")
        return saham_data, latest, summary

    saham_data, latest, summary = load_data()

    # =========================
    # PREPARE CLUSTERING
    # =========================
    df_cluster = prepare_clustering_data(summary)

    X_scaled, scaler = scale_features(df_cluster)

    df_cluster, kmeans = run_kmeans(
        df_cluster,
        X_scaled,
        k=3
    )

    profile = get_cluster_profile(
        df_cluster,
        kmeans,
        scaler
    )

    profile["Label"] = profile.apply(
        interpret_cluster,
        axis=1
    )

    df_cluster = df_cluster.merge(
        profile[["Cluster", "Label"]],
        on="Cluster",
        how="left"
    )

    # =========================
    # SCATTER CHART
    # =========================
    fig1 = px.scatter(
        df_cluster,
        x="Mean_Return",
        y="Risk",
        color="Label",
        hover_name="Stock_Name",
        size="Avg_Volume",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig1.update_traces(
        marker=dict(size=12, line=dict(width=0.5, color="#333"))
    )

    fig1.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cluster Visualization</div>', unsafe_allow_html=True)
    st.caption("Hover untuk melihat nama saham")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # DISTRIBUTION CHART
    # =========================
    cluster_counts = (
        df_cluster["Label"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = ["Cluster", "Count"]

    fig2 = px.bar(
        cluster_counts,
        x="Cluster",
        y="Count",
        text="Count",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig2.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # =========================
    # PROFILE TABLE (STYLE)
    # =========================
    styled_profile = (
        profile.style
        .background_gradient(subset=["Mean_Return"], cmap="Greens")
        .background_gradient(subset=["Risk"], cmap="Reds")
        .background_gradient(subset=["Avg_Volume"], cmap="Blues")
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

    # =========================
    # SIDE-BY-SIDE SECTION
    # =========================
    col1, col2 = st.columns([2, 1])

    # PROFILE TABLE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cluster Profiles</div>', unsafe_allow_html=True)

        st.write(styled_profile.to_html(), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # DISTRIBUTION CHART
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # ASSIGNMENT TABLE
    # =========================
    assignment_df = df_cluster[[
        "Stock_Name",
        "Mean_Return",
        "Risk",
        "Avg_Volume",
        "Label"
    ]].copy()

    assignment_df["Mean_Return"] = assignment_df["Mean_Return"].round(4)
    assignment_df["Risk"] = assignment_df["Risk"].round(4)

    styled_assignment = (
        assignment_df.style
        .background_gradient(subset=["Mean_Return"], cmap="Greens")
        .background_gradient(subset=["Risk"], cmap="Reds")
        .background_gradient(subset=["Avg_Volume"], cmap="Blues")
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Stock Cluster Assignments</div>', unsafe_allow_html=True)

    st.write(styled_assignment.to_html(), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)