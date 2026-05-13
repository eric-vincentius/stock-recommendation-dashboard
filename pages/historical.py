import streamlit as st
import pandas as pd
import plotly.express as px

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
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 20px;
    }

    /* CARD */
    .card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* METRIC */
    [data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="main-title">Historical Stock</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze stock performance and ESG risk</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv("data/saham_data.csv")
    data["Date"] = pd.to_datetime(data["Date"])

    esg_data = pd.read_csv("data/esg_score.csv", sep=";")
    esg_data = esg_data.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    # =========================
    # FILTER + KPI ROW
    # =========================
    col1, col2 = st.columns([2, 1])

    # STOCK SELECT
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        stock = st.selectbox(
            "Select Stock",
            data["Stock_Name"].unique()
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # FILTER DATA
    df_stock = data[data["Stock_Name"] == stock].sort_values("Date")

    df_stock = df_stock.merge(
        esg_data,
        on="Stock_Name",
        how="left"
    )

    esg_score = round(df_stock["esg"].iloc[0], 2)

    # ESG KPI
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.metric(
            label="ESG Risk Score",
            value=esg_score
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # CHART CARD
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(f"{stock} Price Trend")

    fig = px.line(
        df_stock,
        x="Date",
        y="Close"
    )

    fig.update_traces(
        line=dict(width=3)
    )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)