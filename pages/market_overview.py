import streamlit as st
import pandas as pd
from utils.scoring import compute_features
import numpy as np

def show():
    st.title("📊 Market Overview")

    df = pd.read_csv("data/stock_summary.csv")
    df = df.set_index("Stock_Name")

    esg_df = pd.read_csv("data/esg_score.csv", sep=";")
    esg_df = esg_df.rename(columns={"Saham": "Stock_Name", "ESG Score": "esg"})
    esg_df = esg_df.set_index("Stock_Name")  

    df = df.join(esg_df)

    # Compute features
    features = compute_features(df, df["esg"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks", len(features))
    col2.metric("Avg Return", round(features["return"].mean(), 4))
    col3.metric("Avg Risk", round(features["risk"].mean(), 4))
    col4.metric("Avg Volume", int(df["Avg_Volume"].mean()))

    # Display ESG distribution
    conditions = [
        features["esg"] <= 10,
        features["esg"] <= 20,
        features["esg"] <= 30,
        features["esg"] <= 40,
    ]

    choices = ["Negligible Risk", "Low Risk", "Moderate Risk", "High Risk"]

    features["ESG_Category"] = np.select(
        conditions,
        choices,
        default="Severe Risk"
    )

    st.subheader("ESG Risk Distribution")
    esg_counts = features["ESG_Category"].value_counts()
    st.bar_chart(esg_counts)

    # Display sorted by score)
    st.dataframe(features.sort_values("score", ascending=False))