import streamlit as st
import pandas as pd
import plotly.express as px

def show():

    # =========================================================
    # PAGE CONFIG
    # =========================================================
    st.set_page_config(
        page_title="Trending Stocks",
        layout="wide"
    )

    # =========================================================
    # CSS
    # =========================================================
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]{
        font-family: 'Poppins', sans-serif;
    }

    .stApp{
        background:#F4F6F9;
    }

    .block-container{
        padding-top:2rem;
        padding-left:2rem;
        padding-right:2rem;
        max-width:100%;
    }

    /* =========================================================
    TITLE
    ========================================================= */

    .main-title{
        font-size:34px;
        font-weight:800;
        color:#1E293B;
        margin-bottom:0;
    }

    .subtitle{
        color:#64748B;
        font-size:15px;
        margin-bottom:30px;
    }

    /* =========================================================
    METRIC CARD
    ========================================================= */

    div[data-testid="metric-container"]{

        background:white;

        border:2px solid #12411d;

        border-radius:20px;

        padding:18px;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="metric-container"] label{

        color:#64748B !important;

        font-size:24px !important;

        font-weight:800 !important;
    }

    div[data-testid="stMetricValue"]{

        color:#111827 !important;

        font-size:30px !important;

        font-weight:800 !important;
    }

    div[data-testid="stMetricDelta"]{

        color:#16A34A !important;

        font-size:13px !important;
    }

    /* =========================================================
    CHART CARD
    ========================================================= */

    div[data-testid="stPlotlyChart"]{

        background:white;

        border:2px solid #12411d;

        border-radius:20px;

        padding:18px;

        overflow:hidden;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);

        margin-top:10px;
    }

    /* =========================================================
    TABLE CARD
    ========================================================= */

    div[data-testid="stDataFrame"]{

        background:white;

        border:2px solid #12411d;

        border-radius:20px;

        padding:10px;

        overflow:hidden;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    /* =========================================================
    TABLE STYLE
    ========================================================= */

    thead tr th{

        background:#12411d !important;

        color:white !important;

        font-weight:700 !important;

        text-align:center !important;
    }

    tbody tr:nth-child(even){

        background:#F8FAFC !important;
    }

    tbody tr:hover{

        background:#DCFCE7 !important;
    }

    tbody td{

        color:#111827 !important;

        font-size:14px !important;
    }

    /* =========================================================
    PLOTLY FIX
    ========================================================= */

    div[data-testid="stPlotlyChart"] > div{
        padding-bottom:0rem !important;
        margin-bottom:0rem !important;
    }

    .js-plotly-plot .plot-container{
        padding-bottom:0px !important;
        margin-bottom:0px !important;
    }

    svg.main-svg{
        border-radius:20px;
    }

    /* =========================================================
    HEADERS
    ========================================================= */

    h3{
        color:#1E293B !important;
        font-weight:800 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # TITLE
    # =========================================================
    st.markdown(
        '<div class="main-title">Trending Stocks</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Based on news sentiment and attention</div>',
        unsafe_allow_html=True
    )

    # =========================================================
    # LOAD DATA
    # =========================================================
    df = pd.read_csv("data/news.csv")

    # =========================================================
    # SENTIMENT SCORE
    # =========================================================
    def sentiment_to_score(label):

        return {
            "negative": -1,
            "neutral": 0,
            "positive": 1
        }.get(label, 0)

    df["sentiment_score"] = df["sentiment"].apply(
        sentiment_to_score
    )

    # =========================================================
    # SUMMARY
    # =========================================================
    stock_summary = (
        df.groupby("ticker")
        .agg({
            "sentiment_score": "mean",
            "title": "count"
        })
        .rename(columns={
            "title": "news_count"
        })
        .reset_index()
    )

    stock_summary["sentiment_norm"] = (
        stock_summary["sentiment_score"] + 1
    ) / 2

    stock_summary["attention_norm"] = (
        stock_summary["news_count"] /
        stock_summary["news_count"].max()
    )

    stock_summary["final_score"] = (
        0.6 * stock_summary["sentiment_norm"] +
        0.4 * stock_summary["attention_norm"]
    )

    stock_summary = stock_summary.sort_values(
        "final_score",
        ascending=False
    )

    # =========================================================
    # ALERT
    # =========================================================
    top_stock = stock_summary.iloc[0]

    st.success(
        f"Top Trending: {top_stock['ticker']} | "
        f"Score: {top_stock['final_score']:.2f} | "
        f"News: {top_stock['news_count']}"
    )

    # =========================================================
    # TOP 10 METRICS
    # =========================================================
    st.subheader("Top 10 Trending Stocks")

    top10 = stock_summary.head(10).reset_index(drop=True)

    for row_start in range(0, len(top10), 5):

        cols = st.columns(5)

        for i in range(5):

            if row_start + i < len(top10):

                row = top10.iloc[row_start + i]

                cols[i].metric(
                    label=row["ticker"],
                    value=f"{row['final_score']:.2f}",
                    delta=f"{row['news_count']} news"
                )

    # =========================================================
    # CHART
    # =========================================================
    st.subheader("Sentiment vs Attention")

    fig = px.scatter(
        stock_summary,
        x="sentiment_norm",
        y="attention_norm",
        size="final_score",
        color="final_score",
        hover_name="ticker",
        text="ticker",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(
            line=dict(
                width=1.5,
                color="#12411d"
            )
        )
    )

    fig.update_layout(

        paper_bgcolor="white",
        plot_bgcolor="white",

        height=500,

        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        ),

        font=dict(
            family="Poppins",
            size=13,
            color="#1E293B"
        ),

        xaxis=dict(
            title="Sentiment Score",
            gridcolor="rgba(0,0,0,0.06)"
        ),

        yaxis=dict(
            title="Attention Score",
            gridcolor="rgba(0,0,0,0.06)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================================================
    # TABLE
    # =========================================================
    st.subheader("Full Ranking")

    ranking_df = stock_summary[[
        "ticker",
        "sentiment_score",
        "news_count",
        "final_score"
    ]].copy()

# =========================================================

    # =========================================================
    # INDEX DARI 1
    # =========================================================
    ranking_df.index = range(1, len(ranking_df) + 1)

    # =========================================================
    # ROUND
    # =========================================================
    ranking_df = ranking_df.round(3)

    # =========================================================
    # STYLE TABLE
    # =========================================================
    styled_df = (
        ranking_df.style

        .format({
            "sentiment_score": "{:.6f}",
            "final_score": "{:.3f}"
        })

         .background_gradient(
            subset=["final_score"],
            cmap="Greens"
        )

        .map(
            lambda v: "color: white; font-weight:700;"
            if v >= 0.75
            else "color: #111827;",
            subset=["final_score"]
        )

        .set_table_styles([

            # TABLE FULL WIDTH
            {
                'selector': 'table',
                'props': [
                    ('width', '100%'),
                    ('min-width', '100%'),
                    ('border-collapse', 'collapse'),
                    ('border', '2px solid #12411d'),
                    ('border-radius', '16px'),
                    ('overflow', 'hidden'),
                    ('font-family', 'Poppins')
                ]
            },

            # HEADER
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#12411d'),
                    ('color', 'white'),
                    ('font-weight', '700'),
                    ('font-size', '15px'),
                    ('text-align', 'center'),
                    ('padding', '14px'),
                    ('border', '1px solid #d1d5db')
                ]
            },

            # BODY CELL
            {
                'selector': 'tbody td',
                'props': [
                    ('padding', '12px'),
                    ('text-align', 'center'),
                    ('font-size', '14px'),
                    ('border', '1px solid #e5e7eb'),
                    ('color', '#111827')
                ]
            },

            # INDEX COLUMN
            {
                'selector': 'tbody th',
                'props': [
                    ('padding', '12px'),
                    ('text-align', 'center'),
                    ('font-size', '14px'),
                    ('border', '1px solid #e5e7eb'),
                    ('background-color', '#ffffff'),
                    ('color', '#111827'),
                    ('font-weight', '700')
                ]
            }

        ])
    )

    # =========================================================
    # WRAPPER BIAR FULL LEBAR
    # =========================================================
    st.markdown("""
    <style>
    table {
        width: 100% !important;
    }

    .dataframe {
        width: 100% !important;
    }

    tbody tr:hover {
        background-color: #DCFCE7 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # SHOW TABLE
    # =========================================================
    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )