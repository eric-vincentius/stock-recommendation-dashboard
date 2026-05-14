import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score

from models.clustering import (
    prepare_clustering_data,
    scale_features,
    get_cluster_profile,
    interpret_cluster,
    run_kmeans
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Clustering",
    layout="wide"
)

# =========================
# MAIN PAGE
# =========================
def show():

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>

    .block-container{
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
                  html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp{
        background:#F4F6F9;
    }

    /* MAIN TITLE */
    .main-title{
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0px;
    }

    /* SUBTITLE */
    .subtitle{
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }



    /* METRIC CARD */
    .metric-card{
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #f3f4f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-title{
        color: #6b7280;
        font-size: 14px;
        font-weight: 500;
    }

    .metric-value{
        font-size: 32px;
        font-weight: 700;
        color: #111827;
        margin-top: 5px;
    }

    /* SECTION TITLE */
    .section-title{
        font-size: 20px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
    }

    /* INFO BOX */
    .info-box{
        background: #fafafa;
        border: 1px solid #eeeeee;
        padding: 16px;
        border-radius: 14px;
        color: #4b5563;
        font-size: 14px;
    }

    /* TABLE */
    table{
        width: 100%;
        border-collapse: collapse;
    }

    th{
        background: #f9fafb !important;
        text-align: center !important;
        padding: 10px !important;
        font-size: 13px !important;
    }

    td{
        text-align: center !important;
        padding: 10px !important;
        font-size: 13px !important;
    }

    tbody tr:hover{
        background: #f9fafb;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER
    # =========================
    st.markdown(
        '<div class="main-title">Stock Clustering Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Segmentasi saham berdasarkan return, risk, dan volume</div>',
        unsafe_allow_html=True
    )

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

    # =========================
    # SILHOUETTE SCORE
    # =========================
    sil_score = silhouette_score(
        X_scaled,
        df_cluster["Cluster"]
    )

    # =========================
    # CLUSTER PROFILE
    # =========================
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
    # TOP SUMMARY
    # =========================
    top1, top2, top3, top4 = st.columns([3, 1, 1, 1])

    with top1:
        st.markdown("""
        <div class="card">
            <div class="info-box">
                Clustering membantu mengelompokkan saham dengan karakteristik
                serupa untuk strategi investasi yang lebih tepat.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Stocks</div>
            <div class="metric-value">{len(df_cluster)}</div>
        </div>
        """, unsafe_allow_html=True)

    with top3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Clusters</div>
            <div class="metric-value">{df_cluster['Cluster'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with top4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Silhouette Score</div>
            <div class="metric-value">{sil_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

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
        size_max=30,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig1.update_traces(
        marker=dict(
            line=dict(
                width=0.5,
                color="#333"
            )
        )
    )

    fig1.update_layout(
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Mean_Return",
        yaxis_title="Risk"
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Cluster Visualization</div>',
        unsafe_allow_html=True
    )

    st.caption("Hover untuk melihat nama saham")

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

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
        height=350,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10)
    )

    # =========================
    # PROFILE TABLE
    # =========================
    styled_profile = (
        profile.style
        .background_gradient(
            subset=["Mean_Return"],
            cmap="Greens"
        )
        .background_gradient(
            subset=["Risk"],
            cmap="Reds"
        )
        .background_gradient(
            subset=["Avg_Volume"],
            cmap="Blues"
        )
        .format({
            "Mean_Return": "{:.6f}",
            "Risk": "{:.6f}",
            "Avg_Volume": "{:,.0f}"
        })
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

    # =========================
    # 2 COLUMN LAYOUT
    # =========================
    col1, col2 = st.columns([2, 1])

    # LEFT
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Cluster Profiles</div>',
            unsafe_allow_html=True
        )

        st.write(
            styled_profile.to_html(),
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT
    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Distribution</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # ASSIGNMENT TABLE
    # =========================
    assignment_df = df_cluster[[
        "Stock_Name",
        "Mean_Return",
        "Risk",
        "Avg_Volume",
        "Label",
        "Cluster"
    ]].copy()

    assignment_df["Mean_Return"] = (
        assignment_df["Mean_Return"]
        .round(6)
    )

    assignment_df["Risk"] = (
        assignment_df["Risk"]
        .round(6)
    )

    styled_assignment = (
        assignment_df.style
        .background_gradient(
            subset=["Mean_Return"],
            cmap="Greens"
        )
        .background_gradient(
            subset=["Risk"],
            cmap="Reds"
        )
        .background_gradient(
            subset=["Avg_Volume"],
            cmap="Blues"
        )
        .format({
            "Avg_Volume": "{:,.0f}"
        })
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Stock Cluster Assignments</div>',
        unsafe_allow_html=True
    )

    st.write(
        styled_assignment.to_html(),
        unsafe_allow_html=True
        
    )

    st.markdown('</div>', unsafe_allow_html=True)