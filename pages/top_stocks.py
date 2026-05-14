import streamlit as st
import pandas as pd
import plotly.express as px


def show():

    # =========================
    # CLEAN CSS
    # =========================
    st.markdown("""
    <style>
 @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp{
        background:#F4F6F9;
    }
    .block-container {
        padding-top: 1.5rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 20px;
    }

    .card {
        background: white;
        border: 2px solid #12411d;
        border-radius: 20px;
        padding: 20px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="main-title">Trending Stocks</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Based on news sentiment and attention</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/news.csv")

    # =========================
    # TRANSFORM
    # =========================
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

    stock_summary["sentiment_norm"] = (stock_summary["sentiment_score"] + 1) / 2
    stock_summary["attention_norm"] = stock_summary["news_count"] / stock_summary["news_count"].max()

    stock_summary["final_score"] = (
        0.6 * stock_summary["sentiment_norm"] +
        0.4 * stock_summary["attention_norm"]
    )

    stock_summary = stock_summary.sort_values("final_score", ascending=False)

    # =========================
    # TOP STOCK INSIGHT
    # =========================
    top_stock = stock_summary.iloc[0]

    st.success(
        f" Top Trending: {top_stock['ticker']} "
        f"(Score: {top_stock['final_score']:.2f}, News: {top_stock['news_count']})"
    )

    # =========================
    # TOP 10 METRICS
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    
    st.subheader("Top 10 Trending Stocks")

    cols = st.columns(5)

    for i, (_, row) in enumerate(stock_summary.head(10).iterrows()):
        cols[i % 5].metric(
            label=row["ticker"],
            value=f"{row['final_score']:.2f}",
            delta=f"{row['news_count']} news"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # CHART
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
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

    fig.update_traces(textposition="top center")

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TABLE
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Full Ranking")

    stock_summary = stock_summary.drop(columns=["sentiment_norm", "attention_norm"])

    styled_df = (
        stock_summary.style
        .format({
            "final_score": "{:.3f}",
            "sentiment_norm": "{:.3f}",
            "attention_norm": "{:.3f}"
        })
        .background_gradient(subset=["final_score"], cmap="Greens")
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

    st.write(styled_df.to_html(), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)