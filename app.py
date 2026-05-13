# app.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
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
/* BASE ICON STYLE */
.stRadio label::before {

    content: "";

    width: 24px;
    height: 24px;
    min-width: 24px;

    display: inline-block;

    margin-right: 14px;

    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;

    filter: brightness(0) invert(1);
}


/* =========================
ICON 1
========================= */
.stRadio > div > label:nth-child(1)::before {

    background-image: url("https://img.icons8.com/ios-filled/50/home.png");
}

/* =========================
ICON 2
========================= */
.stRadio > div > label:nth-child(2)::before {

    background-image: url("https://img.icons8.com/ios-filled/50/combo-chart.png");
}

/* =========================
ICON 3
========================= */
.stRadio > div > label:nth-child(3)::before {

    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0OCIgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDMyIDMyIj48cGF0aCBmaWxsPSIjZmZmIiBkPSJtMjEuMyA0LjlsLTQuNS0yLjdjLS4zLS4xLS41LS4yLS44LS4ycy0uNS4xLS43LjJsLTQuNSAyLjdjLS41LjMtLjguOC0uOCAxLjN2NS42YzAgLjUuMyAxIC43IDEuM2w0LjUgMi43Yy4yLjEuNS4yLjcuMnMuNS0uMS43LS4ybDQuNS0yLjdjLjQtLjMuNy0uNy43LTEuM1Y2LjJjLjItLjUtLjEtMS0uNS0xLjNNMjAgMTEuNWwtNCAyLjRsLTQtMi40di01bDQtMi40bDQgMi40em05LjMgNy40bC00LjUtMi43Yy0uMy0uMS0uNS0uMi0uOC0uMnMtLjUuMS0uNy4ybC00LjUgMi43Yy0uNC4zLS43LjctLjcgMS4zdjUuNmMwIC41LjMgMSAuNyAxLjNsNC41IDIuN2MuMi4xLjUuMi43LjJzLjUtLjEuNy0uMmw0LjUtMi43Yy40LS4zLjctLjcuNy0xLjN2LTUuNmMuMS0uNS0uMi0xLS42LTEuM00yOCAyNS41bC00IDIuNGwtNC0yLjR2LTVsNC0yLjRsNCAyLjR6bS0xNC43LTYuNmwtNC41LTIuN2MtLjMtLjEtLjUtLjItLjgtLjJzLS41LjEtLjcuMmwtNC41IDIuN2MtLjUuMy0uOC44LS44IDEuM3Y1LjZjMCAuNS4zIDEgLjcgMS4zbDQuNSAyLjdjLjMuMS41LjIuOC4ycy41LS4xLjctLjJsNC41LTIuN2MuNC0uMy43LS43LjctMS4zdi01LjZjLjEtLjUtLjItMS0uNi0xLjNNMTIgMjUuNWwtNCAyLjRsLTQtMi40di01bDQtMi40bDQgMi40eiIvPjwvc3ZnPg==");
}

/* =========================
ICON 4
========================= */
.stRadio > div > label:nth-child(4)::before {

    background-image: url("https://img.icons8.com/ios-filled/50/artificial-intelligence.png");
}

/* =========================
ICON 5
========================= */
.stRadio > div > label:nth-child(5)::before {

    background-image: url("https://img.icons8.com/ios-filled/50/report-card.png");
}

/* =========================
ICON 6
========================= */
.stRadio > div > label:nth-child(6)::before {

    background-image: url("https://img.icons8.com/ios-filled/50/fire-element.png");
}

/* =========================
ICON HOVER COLOR
========================= */
.stRadio label:hover::before {

    filter:
        brightness(0)
        saturate(100%)
        invert(14%)
        sepia(61%)
        saturate(1834%)
        hue-rotate(194deg)
        brightness(93%)
        contrast(101%);
}

/* =========================
ACTIVE ICON COLOR
========================= */
.stRadio input:checked + div label::before {

    filter:
        brightness(0)
        saturate(100%)
        invert(14%)
        sepia(61%)
        saturate(1834%)
        hue-rotate(194deg)
        brightness(93%)
        contrast(101%);
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
with st.sidebar:

    components.html("""
    <div style="
        display:flex;
        align-items:center;
        gap:7px;
        padding:20px 10px 30px 10px;
    ">

        <img 
            src="https://png.pngtree.com/png-vector/20230318/ourmid/pngtree-data-driven-line-icon-vector-png-image_6656311.png"
            width="70"
            color = "white"
        >

        <div style="
            width:3px;
            height:90px;
            background:white;
            border-radius:10px;
        "></div>

        <div style="
            color:white;
            font-family:Poppins;
            font-weight:700;
            font-size:22px;
            line-height:1.2;
            letter-spacing:1px;
        ">
            MARKET <br>
            STOCK <br>
            DASHBOARD
        </div>

    </div>
    """, height=130)

    page = st.radio(
        "",
        [
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