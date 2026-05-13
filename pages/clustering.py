# pages/clustering.py

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.scoring import compute_features
from models.clustering import (
    prepare_clustering_data,
    scale_features,
    get_cluster_profile,
    interpret_cluster,
    run_kmeans,
    compute_elbow
)


# =========================
# PAGE
# =========================
def show():

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #355e2b,
            #5d824f
        );
    }

    .main-title {
        color: white;
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        font-family: Comic Sans MS;
    }

    .section-title {
        color: white;
        font-size: 30px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
        font-family: Comic Sans MS;
    }

    /* CHART */
    div[data-testid="stPlotlyChart"] {

        background: #f4f4f4;

        border-radius: 35px;

        padding: 10px;

        border: 6px solid #1e4b1d;

        overflow: hidden;

        box-shadow:
            0 0 25px rgba(0,0,0,0.2),
            0 0 35px rgba(144,238,144,0.25);

        margin-bottom: 25px;

    }

    /* TABLE */
    table {

        border-collapse: collapse !important;

        width: 100% !important;

        overflow: hidden;

        border-radius: 25px;

        background: #f4f4f4;

    }

    thead tr th {

        background: #4d7f3f !important;

        color: white !important;

        font-size: 16px !important;

        padding: 14px !important;

        text-align: center !important;

    }

    tbody tr td {

        padding: 12px !important;

        text-align: center !important;

        border-bottom: 1px solid #dddddd !important;

        color: #222 !important;

        font-size: 14px !important;

    }

    tbody tr:hover td {

        background: #e6f3df !important;

    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown(
        '<div class="main-title"> Stock Clustering Analysis</div>',
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
    # SCATTER TITLE
    # =========================
    st.markdown(
        '<div class="section-title">Cluster Visualization</div>',
        unsafe_allow_html=True
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

        text="Stock_Name",

        size="Avg_Volume",

        color_discrete_sequence=[
            "#4d7f3f",
            "#7fb069",
            "#b7d3a8"
        ]
    )

    fig1.update_traces(

        textposition="top center",

        marker=dict(
            line=dict(
                width=2,
                color="#1e4b1d"
            )
        ),

        textfont=dict(
            size=14,
            color="#222222"
        )
    )

    fig1.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        height=650,

        font=dict(
            family="Comic Sans MS",
            size=16,
            color="#222"
        ),

        xaxis=dict(

            title="Mean Return",

            title_font=dict(
                size=22,
                color="#1a1a1a"
            ),

            tickfont=dict(
                size=14,
                color="#333333"
            ),

            gridcolor="rgba(0,0,0,0.08)"
        ),

        yaxis=dict(

            title="Risk",

            title_font=dict(
                size=22,
                color="#1a1a1a"
            ),

            tickfont=dict(
                size=14,
                color="#333333"
            ),

            gridcolor="rgba(0,0,0,0.08)"
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =========================
    # PROFILE TITLE
    # =========================
    st.markdown(
        '<div class="section-title">Cluster Profiles</div>',
        unsafe_allow_html=True
    )

    # =========================
    # PROFILE TABLE
    # =========================
    styled_profile = (

    profile.style

    # =====================
    # GRADIENT
    # =====================
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

    .background_gradient(
        subset=["Cluster"],
        cmap="Greens"
    )

    # =====================
    # LABEL COLUMN
    # =====================
    .set_properties(
        subset=["Label"],
        **{
            "background-color": "#e8f2e2",
            "color": "#1e4b1d",
            "font-weight": "bold",
            "border": "1px solid #dcdcdc"
        }
    )

    # =====================
    # GLOBAL STYLE
    # =====================
    .set_properties(**{
        "text-align": "center",
        "font-size": "14px",
        "background-color": "#f4f4f4",
        "color": "#222222"
    })

    # =====================
    # HEADER STYLE
    # =====================
    .set_table_styles([

        {
            "selector": "th",
            "props": [
                ("background-color", "#4d7f3f"),
                ("color", "white"),
                ("font-size", "16px"),
                ("padding", "12px"),
                ("text-align", "center")
            ]
        }

    ])
)
    st.write(
        styled_profile.to_html(),
        unsafe_allow_html=True
    )

    # =========================
    # DISTRIBUTION TITLE
    # =========================
    st.markdown(
        '<div class="section-title">Cluster Distribution</div>',
        unsafe_allow_html=True
    )

    # =========================
    # BAR DATA
    # =========================
    cluster_counts = (
        df_cluster["Label"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Count"
    ]

    # =========================
    # BAR CHART
    # =========================
    fig2 = px.bar(

        cluster_counts,

        x="Cluster",
        y="Count",

        text="Count",

        color="Cluster",

        color_discrete_sequence=[
            "#4d7f3f",
            "#7fb069",
            "#b7d3a8"
        ]
    )

    fig2.update_traces(

        textposition="outside",

        marker_line_color="#1e4b1d",

        marker_line_width=2
    )

    fig2.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        height=600,

        showlegend=False,

        font=dict(
            family="Comic Sans MS",
            size=16,
            color="#222"
        ),

        xaxis=dict(

            title="Cluster",

            title_font=dict(
                size=22,
                color="#1a1a1a"
            ),

            tickfont=dict(
                size=14,
                color="#333333"
            ),

            gridcolor="rgba(0,0,0,0.08)"
        ),

        yaxis=dict(

            title="Total Stocks",

            title_font=dict(
                size=22,
                color="#1a1a1a"
            ),

            tickfont=dict(
                size=14,
                color="#333333"
            ),

            gridcolor="rgba(0,0,0,0.08)"
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =========================
    # ASSIGNMENT TITLE
    # =========================
    st.markdown(
        '<div class="section-title"> Stock Cluster Assignments</div>',
        unsafe_allow_html=True
    )

    # =========================
    # ASSIGNMENT TABLE
    # =========================
    assignment_df = df_cluster[[
        "Stock_Name",
        "Mean_Return",
        "Risk",
        "Avg_Volume",
        "Label"
    ]]

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

        .set_properties(**{
            "text-align": "center",
            "font-size": "14px"
        })
    )

    st.write(
        styled_assignment.to_html(),
        unsafe_allow_html=True
    )