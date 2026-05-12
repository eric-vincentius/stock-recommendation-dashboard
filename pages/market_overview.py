import streamlit as st
import pandas as pd
from utils.scoring import compute_features
import numpy as np
import plotly.express as px

def show():

    st.title("Market Overview\n")

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    df = df.set_index("Stock_Name")

    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    esg_df = esg_df.rename(
        columns={
            "Saham": "Stock_Name",
            "ESG Score": "esg"
        }
    )

    esg_df = esg_df.set_index("Stock_Name")

    # JOIN ESG
    df = df.join(esg_df)

    # =========================
    # COMPUTE FEATURES
    # =========================
    features = compute_features(df, df["esg"])

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>

    [data-testid="stMetric"] {

        background: #F5F5F5;
        border-radius: 28px;
        padding: 20px;
        border: 3px solid #355E3B;

        box-shadow:
            0 0 25px rgba(28,53,84,0.25);
    }

    [data-testid="stMetricLabel"] {

        color: #1F2D1F !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {

        color: #234A2E !important;
        font-size: 45px !important;
        font-weight: 700 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Stocks",
        len(features)
    )

    col2.metric(
        "Avg Return",
        round(features["return"].mean(), 4)
    )

    col3.metric(
        "Avg Risk",
        round(features["risk"].mean(), 4)
    )

    col4.metric(
        "Avg Volume",
        int(df["Avg_Volume"].mean())
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # ESG CATEGORY
    # =========================
    conditions = [
        features["esg"] <= 10,
        features["esg"] <= 20,
        features["esg"] <= 30,
        features["esg"] <= 40,
    ]

    choices = [
        "Negligible Risk",
        "Low Risk",
        "Moderate Risk",
        "High Risk"
    ]

    features["ESG_Category"] = np.select(
        conditions,
        choices,
        default="Severe Risk"
    )

    # =========================
    # ESG DISTRIBUTION
    # =========================
    st.subheader("ESG Risk Distribution")

    esg_counts = (
        features["ESG_Category"]
        .value_counts()
        .reset_index()
    )

    esg_counts.columns = [
        "Kategori",
        "Jumlah"
    ]

    # =========================
    # CHART
    # =========================
    fig = px.bar(
        esg_counts,
        x="Kategori",
        y="Jumlah",
        color="Kategori",

        color_discrete_sequence=[
            "#7BA89A",
            "#83D483",
            "#355E3B",
            "#1F4D2E",
            "#1C3554"
        ]
    )

    fig.update_layout(

        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#F5F5F5",

        font=dict(
            family="Poppins",
            size=15,
            color="black"
        ),

        xaxis=dict(
            title="Kategori ESG",
            showgrid=False
        ),

        yaxis=dict(
            title="Jumlah Saham",
            gridcolor="rgba(0,0,0,0.08)"
        ),

        showlegend=False,

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )

    # =========================
    # AXIS COLOR
    # =========================
    fig.update_xaxes(
        tickfont=dict(
            color="black",
            size=14
        ),

        title_font=dict(
            color="black",
            size=16
        )
    )

    fig.update_yaxes(
        tickfont=dict(
            color="black",
            size=14
        ),

        title_font=dict(
            color="black",
            size=16
        )
    )

    # =========================
    # BAR STYLE
    # =========================
    fig.update_traces(
        marker_line_width=0
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

       # =========================
    # TABLE
    # =========================
    st.subheader("Stock Ranking")

    ranking_df = (
        features
        .sort_values("score", ascending=False)
        .reset_index()
    )

    styled_df = (
        ranking_df.style
        .set_table_styles([

            {
                'selector': 'th',
                'props': [
                    ('background-color', '#355E3B'),
                    ('color', 'white'),
                    ('font-size', '15px'),
                    ('font-family', 'Poppins'),
                    ('text-align', 'center'),
                    ('border', '1px solid #DADADA')
                ]
            },

            {
                'selector': 'td',
                'props': [
                    ('background-color', 'white'),
                    ('color', 'black'),
                    ('font-size', '14px'),
                    ('border', '1px solid #EAEAEA')
                ]
            }

        ])
        .background_gradient(
            cmap="Greens",
            subset=["score"]
        )
    )

    st.table(styled_df)