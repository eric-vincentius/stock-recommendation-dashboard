# pages/investment_summary.py

import streamlit as st
import numpy as np
import pandas as pd
from utils.portfolio import portfolio_metrics

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

    return df

def show():
    st.title("💼 Investment Summary")

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    esg_df = pd.read_csv("data/esg_score.csv", sep=";")

    # Rename ESG column
    esg_df = esg_df.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    # =========================
    # CLEAN + MERGE
    # =========================
    df["Stock_Name"] = df["Stock_Name"].str.strip()
    esg_df["Stock_Name"] = esg_df["Stock_Name"].str.strip()

    df = df.merge(esg_df, on="Stock_Name", how="left")

    df = compute_green_score(df)
    st.subheader("Top 10 Recommended Stocks")
    st.bar_chart(df.sort_values("Green Score", ascending=False).head(10).set_index("Stock_Name")["Green Score"])

    st.subheader("🌱 Green Score per Stock")

    st.dataframe(
        df.sort_values("Green Score", ascending=False)
    )