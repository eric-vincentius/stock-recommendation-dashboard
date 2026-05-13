import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.scoring import compute_features

def badge_esg(val):
    colors = {
        "Severe Risk": "#ef4444",
        "High Risk": "#f97316",
        "Moderate Risk": "#22c55e",
        "Low Risk": "#84cc16",
        "Negligible Risk": "#9ca3af"
    }

    return f'<span style="background:{colors.get(val, "#ccc")}; color:white; padding:6px 12px; border-radius:12px; font-size:12px; font-weight:600;">{val}</span>'

def show():

    # =========================
    # PAGE TITLE
    # =========================
    st.title("Market Overview")

    st.caption("Ringkasan kondisi pasar dan distribusi risiko ESG")

    # =========================
    # CLEAN DASHBOARD CSS
    # =========================
    st.markdown("""
    <style>

    .block-container {
        padding-top: 1.5rem;
    }

    h1 {
        font-size: 34px !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }

    /* KPI */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }

    /* CARD */
    .card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    # CLEAN COLUMN
    esg_df = esg_df.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    df["Stock_Name"] = df["Stock_Name"].str.strip()
    esg_df["Stock_Name"] = esg_df["Stock_Name"].str.strip()

    # MERGE
    df = df.merge(esg_df, on="Stock_Name", how="left")
    df = df.set_index("Stock_Name")

    # =========================
    # COMPUTE FEATURES
    # =========================
    features = compute_features(df, df["esg"])

    # =========================
    # KPI SECTION
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 22px !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    col1.metric("Total Stocks", len(features))
    col2.metric("Avg Return", round(features["return"].mean(), 4))
    col3.metric("Avg Risk", round(features["risk"].mean(), 4))
    col4.metric("Avg Volume", f"{int(df['Avg_Volume'].mean()):,}")

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
    # ESG DISTRIBUTION DATA
    # =========================
    esg_counts = (
        features["ESG_Category"]
        .value_counts()
        .reset_index()
    )

    esg_counts.columns = ["Kategori", "Jumlah"]

    # =========================
    # CHART SECTION
    # =========================
    st.subheader("ESG Risk Distribution")
    col1, col2 = st.columns([2, 1])

    # BAR CHART
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fig_bar = px.bar(
            esg_counts,
            x="Kategori",
            y="Jumlah",
            color="Kategori",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_bar.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # DONUT CHART
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fig_pie = px.pie(
            esg_counts,
            values="Jumlah",
            names="Kategori",
            hole=0.6
        )

        fig_pie.update_layout(
            margin=dict(t=40, b=10)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # STOCK RANKING TABLE
    # =========================
    st.markdown("""
        <style>

        /* Force full width */
        table {
            width: 100% !important;
        }

        /* Make container full width */
        div[data-testid="stMarkdownContainer"] > table {
            width: 100% !important;
        }

        /* Prevent shrinking */
        .block-container {
            max-width: 100% !important;
        }

        /* Optional: improve spacing */
        th, td {
            padding: 12px;
        }
                
        th {
            text-align: center !important;
        }

        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Stock Ranking")

    ranking_df = (
        features
        .sort_values("score", ascending=False)
        .reset_index()
        .copy()
    )

    # format numbers
    ranking_df["return"] = ranking_df["return"].round(4)
    ranking_df["risk"] = ranking_df["risk"].round(4)
    ranking_df["score"] = ranking_df["score"].round(4)

    # optional: select columns
    ranking_df = ranking_df[
        ["Stock_Name", "return", "risk", "score", "ESG_Category"]
    ]

    # badge styling
    ranking_df["ESG_Category"] = ranking_df["ESG_Category"].fillna("Unknown")
    ranking_df["ESG_Category"] = ranking_df["ESG_Category"].apply(badge_esg)

    # render
    st.markdown("""
    <div style="width: 100%;">
    """, unsafe_allow_html=True)

    st.write(ranking_df.to_html(escape=False), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)