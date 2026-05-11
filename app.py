# app.py
import streamlit as st
import pandas as pd
from clustering import *
from models.clustering import (
    prepare_clustering_data,
    scale_features,
    run_kmeans,
    compute_elbow
)
from models.forecasting import train_lstm, forecast_future

st.set_page_config(layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    saham_data = pd.read_csv("data/saham_data.csv")
    latest = pd.read_csv("data/latest_snapshot.csv")
    summary = pd.read_csv("data/stock_summary.csv")
    return saham_data, latest, summary

saham_data, latest, summary = load_data()

page = st.sidebar.radio("Menu", [
    "Market Overview",
    "Historical Stock",
    "Clustering",
    "Forecasting",
    "Investment Summary",
    "Trending Stocks"
])

if page == "Market Overview":
    from pages.market_overview import show
    show()

elif page == "Historical Stock":
    from pages.historical import show
    show()

elif page == "Clustering":
    from pages.clustering import show
    show()

elif page == "Forecasting":
    from pages.forecasting import show
    show()

elif page == "Investment Summary":
    from pages.investment_summary import show
    show()

elif page == "Trending Stocks":
    from pages.top_stocks import show
    show()