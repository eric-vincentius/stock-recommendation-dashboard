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

        margin-bottom: 15px;
    }

    /* METRIC TEXT */
    div[data-testid="metric-container"] label {

        color: #2e5d34 !important;

        font-size: 24px !important;

        font-weight: bold !important;
    }

    /* METRIC VALUE */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {

        color: #111111 !important;

        font-size: 32px !important;

        font-weight: bold !important;
    }

    /* DELTA */
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {

        color: #4d7f3f !important;

        font-size: 16px !important;
    }

    /* PLOTLY CHART */
    div[data-testid="stPlotlyChart"] {

        background: #f4f4f4;

        border-radius: 30px;

        padding: 10px;

        border: 5px solid #1e4b1d;

        overflow: hidden;

        box-shadow:
            0 0 20px rgba(0,0,0,0.15),
            0 0 30px rgba(144,238,144,0.20);

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
        '<div class="main-title"> Top Trending Stocks</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/news.csv")

    # =========================
    # TRANSFORM
    # =========================
    def sentiment_to_score(label):
        return {
            "negative": -1,
            "neutral": 0,
            "positive": 1
        }.get(label, 0)

    df["sentiment_score"] = df["sentiment"].apply(sentiment_to_score)

    stock_summary = df.groupby("ticker").agg({
        "sentiment_score": "mean",
        "title": "count"
    }).rename(columns={
        "title": "news_count"
    }).reset_index()

    # =========================
    # NORMALIZE
    # =========================
    stock_summary["sentiment_norm"] = (
        stock_summary["sentiment_score"] + 1
    ) / 2

    stock_summary["attention_norm"] = (
        stock_summary["news_count"] /
        stock_summary["news_count"].max()
    )

    # =========================
    # FINAL SCORE
    # =========================
    stock_summary["final_score"] = (
        0.6 * stock_summary["sentiment_norm"] +
        0.4 * stock_summary["attention_norm"]
    )

    # =========================
    # SORT
    # =========================
    stock_summary = stock_summary.sort_values(
        "final_score",
        ascending=False
    )

    # =========================
    # TOP 10
    # =========================
    st.markdown(
        '<div class="section-title"> Top 10 Stocks</div>',
        unsafe_allow_html=True
    )

    top10 = stock_summary.head(10)

    col1, col2 = st.columns(2)

    for i, (_, row) in enumerate(top10.iterrows()):

        target_col = col1 if i % 2 == 0 else col2

        with target_col:

            st.metric(
                label=row["ticker"],
                value=round(row["final_score"], 3),
                delta=f"News: {row['news_count']}"
            )

    # =========================
    # TABLE TITLE
    # =========================
    st.markdown(
        '<div class="section-title"> Full Ranking</div>',
        unsafe_allow_html=True
    )

    # =========================
    # TABLE STYLE
    # =========================
    styled_df = (

        stock_summary.style

        .format({
            "sentiment_score": "{:.3f}",
            "sentiment_norm": "{:.3f}",
            "attention_norm": "{:.3f}",
            "final_score": "{:.3f}"
        })

        .background_gradient(
            subset=["final_score"],
            cmap="Greens"
        )

        .background_gradient(
            subset=["sentiment_norm"],
            cmap="Blues"
        )

        .background_gradient(
            subset=["attention_norm"],
            cmap="Oranges"
        )
    )

    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )

    # =========================
    # CHART TITLE
    # =========================
    st.markdown(
        '<div class="section-title"> Sentiment vs Attention</div>',
        unsafe_allow_html=True
    )

    # =========================
    # SCATTER CHART
    # =========================
    fig = px.scatter(
        stock_summary,
        x="sentiment_norm",
        y="attention_norm",
        size="final_score",
        color="final_score",
        hover_name="ticker",
        text="ticker",
        color_continuous_scale="Greens"
    )

    fig.update_traces(
        textposition="top center"
    )

    fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    height=650,

    font=dict(
        family="Comic Sans MS",
        size=16,
        color="#222222"
    ),

    xaxis=dict(

        title="Sentiment Score",

        title_font=dict(
            size=22,
            color="#1a1a1a"
        ),

        tickfont=dict(
            size=14,
            color="#333333"
        ),

        gridcolor="rgba(0,0,0,0.10)",

        zeroline=False
    ),

    yaxis=dict(

        title="Attention Score",

        title_font=dict(
            size=22,
            color="#1a1a1a"
        ),

        tickfont=dict(
            size=14,
            color="#333333"
        ),

        gridcolor="rgba(0,0,0,0.10)",

        zeroline=False
    ),

    coloraxis_colorbar=dict(

        title=dict(
            text="Final Score",
            font=dict(
                size=18,
                color="#1a1a1a"
            )
        ),

        tickfont=dict(
            size=13,
            color="#333333"
        )

    )
)
    st.plotly_chart(
        fig,
        use_container_width=True
    )