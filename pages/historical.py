import streamlit as st
import pandas as pd
import plotly.express as px

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

    /* SELECTBOX */
    div[data-baseweb="select"] > div {

        background: #f4f4f4 !important;

        border: 4px solid #1e4b1d !important;

        border-radius: 18px !important;

        color: #222 !important;

        box-shadow:
            0 0 15px rgba(0,0,0,0.15),
            0 0 20px rgba(144,238,144,0.15);

    }

    /* METRIC CARD */
    div[data-testid="metric-container"] {

        background: linear-gradient(
            135deg,
            #f4f4f4,
            #e8f2e2
        );

        border: 5px solid #1e4b1d;

        padding: 18px;

        border-radius: 25px;

        box-shadow:
            0 0 20px rgba(0,0,0,0.15),
            0 0 25px rgba(144,238,144,0.20);

        margin-bottom: 20px;
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

    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown(
        '<div class="main-title">Historical Stock</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv("data/saham_data.csv")

    # =========================
    # DATE FORMAT
    # =========================
    data["Date"] = pd.to_datetime(data["Date"])

    # =========================
    # STOCK SELECT
    # =========================
    st.markdown(
        '<div class="section-title"> Select Stock</div>',
        unsafe_allow_html=True
    )

    stock = st.selectbox(
        "",
        data["Stock_Name"].unique()
    )

    # =========================
    # FILTER STOCK
    # =========================
    df_stock = (
        data[data["Stock_Name"] == stock]
        .sort_values("Date")
    )

    # =========================
    # ESG DATA
    # =========================
    esg_data = pd.read_csv(
        "data/esg_score.csv",
        sep=";"
    )

    esg_data = esg_data.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    esg_data["Stock_Name"] = (
        esg_data["Stock_Name"]
        .str.strip()
    )

    df_stock["Stock_Name"] = (
        df_stock["Stock_Name"]
        .str.strip()
    )

    df_stock = df_stock.merge(
        esg_data,
        on="Stock_Name",
        how="left"
    )

    # =========================
    # ESG SCORE CARD
    # =========================
    esg_score = round(
        df_stock["esg"].iloc[0],
        2
    )

    st.metric(
        label=f"{stock} ESG Risk",
        value=esg_score
    )

    # =========================
    # CHART TITLE
    # =========================
    st.markdown(
        f'<div class="section-title"> {stock} Closing Price</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LINE CHART
    # =========================
    fig = px.line(
        df_stock,
        x="Date",
        y="Close"
    )

    fig.update_traces(

        line=dict(
            color="#4d7f3f",
            width=4
        )

    )

    fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    height=650,

    font=dict(
        family="Comic Sans MS",
        size=16,
        color="#222"
    ),

    xaxis=dict(

        title="Date",

        title_font=dict(
            size=22,
            color="#1a1a1a"
        ),

        tickfont=dict(
            size=14,
            color="#333333"
        ),

        gridcolor="rgba(0,0,0,0.08)",

        showline=False,

        zeroline=False
    ),

    yaxis=dict(

        title="Closing Price",

        title_font=dict(
            size=22,
            color="#1a1a1a"
        ),

        tickfont=dict(
            size=14,
            color="#333333"
        ),

        gridcolor="rgba(0,0,0,0.08)",

        showline=False,

        zeroline=False
    )
)
    st.plotly_chart(
        fig,
        use_container_width=True
    )