import streamlit as st
import pandas as pd
import plotly.express as px

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

    df = df.drop(columns=["NormESG", "NormReturn", "NormRisk"])
    df = df.drop(columns=["Avg_ATR_Pct", "Worst_Drawdown60"])

    return df


# =========================
# PAGE
# =========================
def show():

    # =========================
    # GLOBAL CLEAN CSS
    # =========================
    st.markdown("""
    <style>

    .stApp {
        background: #f5f7fa;
    }

    .main-title {
        color: #1e293b;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 25px;
    }

    .section-title {
        color: #334155;
        font-size: 22px;
        font-weight: 700;
        margin: 25px 0 10px 0;
    }

    /* CARD */
    .card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* CHART */
    div[data-testid="stPlotlyChart"] {
        background: white;
        border-radius: 18px;
        padding: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    /* TABLE */
    table {
        width: 100% !important;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
    }

    thead th {
        background: #f1f5f9 !important;
        color: #334155 !important;
        font-weight: 600;
        text-align: center !important;
        padding: 10px;
    }

    tbody td {
        text-align: center;
        padding: 10px;
        border-bottom: 1px solid #eee;
    }

    tbody tr:hover td {
        background: #f9fafb;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="main-title">Investment Summary</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    esg_df = esg_df.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    df["Stock_Name"] = df["Stock_Name"].str.strip()
    esg_df["Stock_Name"] = esg_df["Stock_Name"].str.strip()

    df = df.merge(esg_df, on="Stock_Name", how="left")

    # =========================
    # SCORE
    # =========================
    df = compute_green_score(df)

    # =========================
    # TOP 10
    # =========================
    top_df = df.sort_values("Green Score", ascending=False).head(10)

    # =========================
    # CHART
    # =========================
    st.markdown('<div class="section-title">Top Performing Stocks</div>', unsafe_allow_html=True)

    st.caption("Green score dihitung berdasarkan ESG, return, dan risiko")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    fig = px.bar(
        top_df,
        x="Stock_Name",
        y="Green Score",
        text_auto=".2f",
        color="Green Score",
        color_continuous_scale="Greens"
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#334155"),
        xaxis=dict(title="Stock"),
        yaxis=dict(title="Score"),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TABLE
    # =========================
    st.markdown('<div class="section-title">Full Ranking</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    ranking_df = df.sort_values("Green Score", ascending=False).reset_index(drop=True)

    styled_df = (
        ranking_df.style
        .format({
            "Avg_Return": "{:.4f}",
            "Return_Volatility": "{:.4f}",
            "Avg_Volume": "{:,.0f}",
            "ESG Risk": "{:.2f}",
            "Green Score": "{:.4f}"
        })
        .background_gradient(subset=["Green Score"], cmap="Greens")
        .background_gradient(subset=["Avg_Return"], cmap="Blues")
        .background_gradient(subset=["Return_Volatility"], cmap="Oranges")
    )

    st.write(styled_df.to_html(), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)