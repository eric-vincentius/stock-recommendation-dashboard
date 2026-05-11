import streamlit as st
import pandas as pd

def show():
    st.title("🔥 Top Trending Stocks")

    # Load your data
    df = pd.read_csv("data/news.csv")

    # -------------------------
    # Transform
    # -------------------------
    def sentiment_to_score(label):
        return {"negative": -1, "neutral": 0, "positive": 1}.get(label, 0)

    df["sentiment_score"] = df["sentiment"].apply(sentiment_to_score)

    stock_summary = df.groupby("ticker").agg({
        "sentiment_score": "mean",
        "title": "count"
    }).rename(columns={"title": "news_count"}).reset_index()

    # Normalize
    stock_summary["sentiment_norm"] = (stock_summary["sentiment_score"] + 1) / 2
    stock_summary["attention_norm"] = stock_summary["news_count"] / stock_summary["news_count"].max()

    # Final score
    stock_summary["final_score"] = (
        0.6 * stock_summary["sentiment_norm"] +
        0.4 * stock_summary["attention_norm"]
    )

    # Sort
    stock_summary = stock_summary.sort_values("final_score", ascending=False)

    # -------------------------
    # UI
    # -------------------------
    st.subheader("🏆 Top 10 Stocks")

    top10 = stock_summary.head(10)

    for _, row in top10.iterrows():
        st.metric(
            label=row["ticker"],
            value=round(row["final_score"], 3),
            delta=f"News: {row['news_count']}"
        )

    # -------------------------
    # Table
    # -------------------------
    st.subheader("📊 Full Ranking")
    st.dataframe(stock_summary)

    # -------------------------
    # Chart
    # -------------------------
    st.subheader("📈 Sentiment vs Attention")
    st.scatter_chart(
        stock_summary.set_index("ticker")[["sentiment_norm", "attention_norm"]]
    )