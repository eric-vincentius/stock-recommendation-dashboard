import streamlit as st
import pandas as pd
import plotly.express as px
import math


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
    df = df.drop(columns=["Avg_ATR_Pct", "Worst_Drawdown60"], errors="ignore")

    return df


# =========================
# PAGE
# =========================
def show():

    st.markdown("""
    <style>

    /* Background */
    .stApp {
        background: #f0f2f5;
    }

    /* Page title */
    .main-title {
        color: #1e293b;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 20px;
        padding-top: 10px;
    }

    /* Card — matches screenshot border */
    .card {
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        margin-bottom: 16px;
    }

    .card-title {
        color: #1e293b;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .card-caption {
        color: #94a3b8;
        font-size: 12px;
    }

    /* Plotly chart */
    div[data-testid="stPlotlyChart"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 10px;
        margin-bottom: 16px;
    }

    /* Table card */
    .table-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 16px 20px 20px 20px;
        margin-bottom: 12px;
    }

    .table-card-title {
        color: #1e293b;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    /* Table */
    table {
        width: 100% !important;
        border-collapse: collapse;
        font-size: 12.5px;
    }

    thead th {
        background: #f8fafc !important;
        color: #475569 !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding: 9px 10px !important;
        border-top: 1px solid #e2e8f0 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        white-space: nowrap;
    }

    tbody td {
        text-align: center !important;
        padding: 8px 10px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        color: #334155;
    }

    tbody tr:hover td {
        background: #f9fafb !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # ── Title ──
    st.markdown('<div class="main-title">Investment Summary</div>', unsafe_allow_html=True)

    # ── Load Data ──
    df = pd.read_csv("data/stock_summary.csv")
    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    esg_df = esg_df.rename(columns={"Saham": "Stock_Name", "ESG Score": "esg"})
    df["Stock_Name"] = df["Stock_Name"].str.strip()
    esg_df["Stock_Name"] = esg_df["Stock_Name"].str.strip()

    df = df.merge(esg_df, on="Stock_Name", how="left")
    df = compute_green_score(df)

    top_df = df.sort_values("Green Score", ascending=False).head(10)

    # ── Card: section header ──
    st.markdown("""
    <div class="card">
        <div class="card-title">Top Performing Stocks</div>
        <div class="card-caption">Green score dihitung berdasarkan ESG, return, dan risiko</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ──
    fig = px.bar(
        top_df,
        x="Stock_Name",
        y="Green Score",
        text_auto=".2f",
        color="Green Score",
        color_continuous_scale=[
            [0.0,  "#c8e6c9"],
            [0.35, "#81c784"],
            [0.65, "#388e3c"],
            [1.0,  "#1b5e20"],
        ],
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11.5, color="#334155"),
        marker_line_width=0,
    )

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#475569", size=12),
        xaxis=dict(
            title="Stock",
            tickfont=dict(size=11),
            showgrid=False,
            linecolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(
            title="Score",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            range=[0, top_df["Green Score"].max() * 1.18],
        ),
        coloraxis_colorbar=dict(
            title="Green Score",
            tickformat=".2f",
            thickness=14,
            len=0.75,
            outlinewidth=0,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Full Ranking — prepare ──
    ranking_df = (
        df.sort_values("Green Score", ascending=False)
        .reset_index(drop=True)
    )

    # "#" column starts from 0
    ranking_df.insert(0, "#", range(len(ranking_df)))

    display_cols = ["#", "Stock_Name", "Avg_Return", "Return_Volatility",
                    "Avg_Volume", "esg", "Green Score"]
    ranking_df = ranking_df[display_cols].copy()

    # ── Pagination state ──
    ROWS_PER_PAGE = 15
    total_rows = len(ranking_df)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)

    if "invest_page" not in st.session_state:
        st.session_state["invest_page"] = 1

    current_page = st.session_state["invest_page"]
    start_idx = (current_page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_df = ranking_df.iloc[start_idx:end_idx]

    # ── Table card ──
    st.markdown('<div class="table-card"><div class="table-card-title">Full Ranking</div>', unsafe_allow_html=True)

    styled = (
        page_df.style
        .format({
            "#": "{}",
            "Avg_Return": "{:.4f}",
            "Return_Volatility": "{:.4f}",
            "Avg_Volume": "{:,.0f}",
            "esg": "{:.6f}",
            "Green Score": "{:.4f}",
        })
        .background_gradient(subset=["Green Score"], cmap="Greens")
        .background_gradient(subset=["Avg_Return"], cmap="Blues")
        .background_gradient(subset=["Return_Volatility"], cmap="Oranges")
        .set_properties(**{"text-align": "center"})
        .hide(axis="index")
    )

    st.write(styled.to_html(), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Pagination controls ──
    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("← Prev", disabled=(current_page == 1), use_container_width=True):
            st.session_state["invest_page"] -= 1
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; color:#64748b; font-size:13px; padding-top:8px;'>"
            f"Page {current_page} of {total_pages} &nbsp;·&nbsp; {total_rows} stocks</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button("Next →", disabled=(current_page == total_pages), use_container_width=True):
            st.session_state["invest_page"] += 1
            st.rerun()