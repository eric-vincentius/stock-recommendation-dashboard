# pages/investment_summary.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils.portfolio import portfolio_metrics


# =========================
# COMPUTE GREEN SCORE
# =========================
def compute_green_score(df):

    df["NormESG"] = (100 - df["esg"]) / 100

    df["NormReturn"] = (
        (df["Avg_Return"] - df["Avg_Return"].min()) /
        (df["Avg_Return"].max() - df["Avg_Return"].min())
    )

    df["NormRisk"] = (
        (df["Return_Volatility"] - df["Return_Volatility"].min()) /
        (df["Return_Volatility"].max() - df["Return_Volatility"].min())
    )

    df["Green Score"] = (
        0.5 * df["NormESG"] +
        0.3 * df["NormReturn"] +
        0.2 * (1 - df["NormRisk"])
    )

    return df


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
    }

    div[data-testid="stPlotlyChart"] {

        background: #f4f4f4;

        border-radius: 35px;

        padding: 10px;

        border: 6px solid #1e4b1d;

        overflow: hidden;

        box-shadow:
            0 0 25px rgba(0,0,0,0.2),
            0 0 35px rgba(144,238,144,0.25);

    }

    div[data-testid="stDataFrame"] {

        background: #f4f4f4;

        border-radius: 30px;

        padding: 12px;

        border: 5px solid #1e4b1d;

        overflow: hidden;

        box-shadow:
            0 0 20px rgba(0,0,0,0.15),
            0 0 30px rgba(144,238,144,0.20);

    }
                
                table {

    border-collapse: collapse !important;

    width: 100% !important;

    overflow: hidden;

    border-radius: 25px;

    background: #12411d;

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
        '<div class="main-title">Investment Summary</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    # =========================
    # RENAME ESG COLUMN
    # =========================
    esg_df = esg_df.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    # =========================
    # CLEAN DATA
    # =========================
    df["Stock_Name"] = df["Stock_Name"].str.strip()
    esg_df["Stock_Name"] = esg_df["Stock_Name"].str.strip()

    # =========================
    # MERGE
    # =========================
    df = df.merge(
        esg_df,
        on="Stock_Name",
        how="left"
    )

    # =========================
    # COMPUTE SCORE
    # =========================
    df = compute_green_score(df)

    # =========================
    # TOP 10
    # =========================
    top_df = (
        df.sort_values(
            "Green Score",
            ascending=False
        )
        .head(10)
    )

    # =========================
    # BAR CHART
    # =========================
    fig = px.bar(
        top_df,
        x="Stock_Name",
        y="Green Score",
        text_auto=".2f"
    )

    # =========================
    # BAR STYLE
    # =========================
    fig.update_traces(

        marker_color="#5f9c4d",

        textposition="outside",

        hovertemplate=
        "<b>%{x}</b><br>" +
        "Green Score: %{y:.2f}" +
        "<extra></extra>"
    )

    # =========================
    # CHART LAYOUT
    # =========================
    fig.update_layout(

        title=dict(
            text="Saham Berperforma Terbaik",
            x=0.5,
            xanchor="center",

            font=dict(
                size=38,
                family="Comic Sans MS",
                color="#1a1a1a"
            )
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        height=650,

        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20
        ),

        xaxis=dict(

            title="Stock Name",

            title_font=dict(
                size=28,
                family="Comic Sans MS",
                color="#111111"
            ),

            tickfont=dict(
                size=18,
                family="Comic Sans MS",
                color="#5c4ce0"
            ),

            showgrid=False,
            zeroline=False
        ),

        yaxis=dict(

            title="Green Score Normalized",

            title_font=dict(
                size=28,
                family="Comic Sans MS",
                color="#111111"
            ),

            tickfont=dict(
                size=18,
                family="Comic Sans MS",
                color="#5c4ce0"
            ),

            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False
        )
    )

    # =========================
    # SHOW CHART
    # =========================
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # TABLE TITLE
    # =========================
    st.markdown("""
    <h2 style="
        color:white;
        font-family: Comic Sans MS;
        margin-top:20px;
    ">
 Green Score per Stock
    </h2>
    """, unsafe_allow_html=True)

    # =========================
    # RANKING DATA
    # =========================
    ranking_df = (
        df.sort_values(
            "Green Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # =========================
    # TABLE STYLE
    # =========================
    styled_df = (

        ranking_df.style

        # FORMAT NUMBER
        .format({
            "Avg_Return": "{:.4f}",
            "Return_Volatility": "{:.4f}",
            "Avg_ATR_Pct": "{:.4f}",
            "Worst_Drawdown60": "{:.4f}",
            "Avg_Volume": "{:,.0f}",
            "esg": "{:.2f}",
            "NormESG": "{:.4f}",
            "NormReturn": "{:.4f}",
            "NormRisk": "{:.4f}",
            "Green Score": "{:.4f}"
        })

        # COLUMN COLORS
        .background_gradient(
            subset=["Green Score"],
            cmap="Greens"
        )

        .background_gradient(
            subset=["Avg_Return"],
            cmap="Blues"
        )

        .background_gradient(
            subset=["Return_Volatility"],
            cmap="Oranges"
        )

        .background_gradient(
            subset=["NormESG"],
            cmap="Purples"
        )

        .background_gradient(
            subset=["NormReturn"],
            cmap="BuGn"
        )

        .background_gradient(
            subset=["NormRisk"],
            cmap="Reds"
        )

        # GLOBAL STYLE
        .set_properties(**{
            "background-color": "#f4f4f4",
            "color": "#222222",
            "border-color": "#dcdcdc",
            "font-size": "14px",
            "text-align": "center"
        })

        # TABLE STYLE
        .set_table_styles([

            # HEADER
            {
                "selector": "th",
                "props": [
                    ("background-color", "#4d7f3f"),
                    ("color", "white"),
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "none"),
                    ("padding", "12px")
                ]
            },

            # BODY
            {
                "selector": "td",
                "props": [
                    ("padding", "10px"),
                    ("border-bottom", "1px solid #dddddd")
                ]
            },

            # HOVER
            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#e7f3df")
                ]
            }

        ])
    )

    # =========================
    # SHOW TABLE
    # =========================
    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )