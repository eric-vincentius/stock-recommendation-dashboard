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
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">

<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>

/* GLOBAL FONT */
html, body, [class*="css"]  {

    font-family: 'Poppins', sans-serif;

}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">

<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Background utama */
.stApp {

    background:
        linear-gradient(
            135deg,
            #355e3b,
            #5f8d4e,
            #1f4037
        );

}


</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* =========================
SIDEBAR BACKGROUND
========================= */
[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #143d2a,
            #355e3b,
            #1c3554
        );

    padding-top: 30px;

    border-right:
        2px solid rgba(255,255,255,0.08);
}

/* =========================
MENU CONTAINER
========================= */
.stRadio > div {

    gap: 12px;
}

/* =========================
MENU DEFAULT
========================= */
.stRadio label {

    display: flex;

    align-items: center;

    font-family: 'Cinzel', serif !important;

    font-size: 19px !important;

    font-weight: 700 !important;

    text-transform: uppercase;

    letter-spacing: 1px;

    color: white !important;

    background: transparent !important;

    padding: 18px 20px;

    border-radius: 22px;

    margin-bottom: 10px;

    transition: all 0.3s ease;

    border: 1px solid transparent;

    box-shadow: none;
}
/* =========================
TEXT COLOR
========================= */
.stRadio label p {

    color: white !important;
}

/* =========================
HOVER
========================= */
.stRadio label:hover {

    background:
        rgba(255,255,255,0.92) !important;

    transform: translateX(6px);

    border:
        1px solid rgba(255,255,255,0.2);

    box-shadow:
        0 0 25px rgba(255,255,255,0.18);

}

/* HOVER TEXT */
.stRadio label:hover p {

    color: #263b63 !important;
}

/* =========================
ACTIVE MENU
========================= */
.stRadio input:checked + div label {

    background:
        rgba(255,255,255,0.92) !important;

    border-radius: 22px;

    padding: 18px 20px;

    border:
        1px solid rgba(255,255,255,0.2);

    box-shadow:
        0 0 25px rgba(255,255,255,0.18);
}

/* ACTIVE TEXT */
.stRadio input:checked + div label p {

    color: #263b63 !important;

    font-weight: 900 !important;
}

/* =========================
REMOVE RADIO DOT
========================= */


/* =========================
REMOVE EMPTY LABEL
========================= */
.stRadio > label {

    display: none;
}
/* HIDE RADIO DOT */


/* =========================
HIDE DEFAULT RADIO DOT
========================= */
/* HIDE DOT ONLY */
.stRadio label > div:first-child {

    display: none !important;
    
}

/* =========================
CUSTOM ICON
========================= */
.stRadio label::before {

    content: "";

    width: 24px;

    height: 24px;

    min-width: 24px;

    display: inline-block;

    margin-right: 14px;

    background-image:
    url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0OCIgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDIwNDggMjA0OCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTI1NiAxNDA4di0yNTZIMFYwaDE1MzZ2MjU2aDI1NnY1MTJoLTEyOFYzODRIMzg0djg5Nmg3Njh2MTI4em0wLTM4NFYyNTZoMTE1MlYxMjhIMTI4djg5NnptMjU2IDEyOFY4OTZoMTI4djI1NnptMjU2IDBWNjQwaDEyOHY1MTJ6bTI1NiAwVjc2OGgxMjh2Mzg0em02NDAtMjU2cTc5IDAgMTQ5IDMwdDEyMiA4MnQ4MyAxMjN0MzAgMTQ5cTAgODAtMzAgMTQ5dC04MiAxMjJ0LTEyMyA4M3QtMTQ5IDMwcS02MCAwLTExNy0xOHQtMTA1LTUzbC00MzcgNDM2cS0xOSAxOS00NSAxOXQtNDUtMTl0LTE5LTQ1dDE5LTQ1bDQzNi00MzdxLTM1LTQ4LTUzLTEwNXQtMTgtMTE3cTAtNzkgMzAtMTQ5dDgyLTEyMnQxMjItODN0MTUwLTMwbTAgNjQwcTUzIDAgOTktMjB0ODItNTV0NTUtODF0MjAtMTAwcTAtNTMtMjAtOTl0LTU1LTgydC04MS01NXQtMTAwLTIwcS01MyAwLTk5IDIwdC04MiA1NXQtNTUgODF0LTIwIDEwMHEwIDUzIDIwIDk5dDU1IDgydDgxIDU1dDEwMCAyMCIvPjwvc3ZnPg==");

    background-size: contain;

    background-repeat: no-repeat;

    background-position: center;

    filter: brightness(0) invert(1);
}

</style>
""", unsafe_allow_html=True)
# Load Bootstrap 4 CSS
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-xOolHFLEh07PJGoPkLv1IbcEPTNtaed2xpHsD9ESMhqIYd0nLMwNLD69Npy4HI+N" crossorigin="anonymous">', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    saham_data = pd.read_csv("data/saham_data.csv")
    latest = pd.read_csv("data/latest_snapshot.csv")
    summary = pd.read_csv("data/stock_summary.csv")
    return saham_data, latest, summary

saham_data, latest, summary = load_data()

page = st.sidebar.radio(
    label="",
    options=[
        "MARKET OVERVIEW",
        "HISTORICAL STOCK",
        "CLUSTERING",
        "FORECASTING",
        "INVESTMENT SUMMARY",
        "TRENDING STOCKS"
    ]
)

if page == "MARKET OVERVIEW":
    from pages.market_overview import show
    show()

elif page == "HISTORICAL STOCK":
    from pages.historical import show
    show()

elif page == "CLUSTERING":
    from pages.clustering import show
    show()

elif page == "FORECASTING":
    from pages.forecasting import show
    show()

elif page == "INVESTMENT SUMMARY":
    from pages.investment_summary import show
    show()

elif page == "TRENDING STOCKS":
    from pages.top_stocks import show
    show()