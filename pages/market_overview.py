import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.scoring import compute_features


st.set_page_config(
    page_title="Market Overview",
    layout="wide"
)


def show():

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/stock_summary.csv")
    df = df.set_index("Stock_Name")

    esg_df = pd.read_csv(
        "data/esg_score.csv",
        sep=";"
    )

    esg_df = esg_df.rename(
        columns={
            "Saham": "Stock_Name",
            "ESG Score": "esg"
        }
    )

    esg_df = esg_df.set_index("Stock_Name")

    df = df.join(esg_df)

    # =========================
    # FEATURES
    # =========================
    features = compute_features(
        df,
        df["esg"]
    )

    # =========================
    # ESG CATEGORY
    # =========================
    conditions = [
        features["esg"] <= 10,
        features["esg"] <= 20,
        features["esg"] <= 30,
        features["esg"] <= 40,
    ]

    choices = [
        "Negligible Risk",
        "Low Risk",
        "Moderate Risk",
        "High Risk"
    ]

    features["ESG_Category"] = np.select(
        conditions,
        choices,
        default="Severe Risk"
    )

    # =========================
    # ESG COUNTS
    # =========================
    esg_counts = (
        features["ESG_Category"]
        .value_counts()
        .reset_index()
    )

    esg_counts.columns = [
        "Kategori",
        "Jumlah"
    ]

    # =========================
    # CSS
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

    .block-container{
        padding-top:2rem;
        padding-left:2rem;
        padding-right:2rem;
        max-width:100%;
    }

    h1{
        color:#1E293B !important;
        font-weight:800 !important;
        margin-bottom:0 !important;
    }

    .subtitle{
        color:#64748B;
        font-size:15px;
        margin-bottom:30px;
    }

    div[data-testid="stMetric"]{

        background:white;

        border:2px solid #12411d;

        border-radius:24px;

        padding:22px;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricLabel"]{
        color:#64748B;
        font-size:14px;
        font-weight:600;
    }

    div[data-testid="stMetricValue"]{
        color:#111827;
        font-size:34px;
        font-weight:800;
    }

    div[data-testid="stPlotlyChart"]{

        background:white;

        border:2px solid #12411d;

        border-radius:24px;

        padding:12px;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="stDataFrame"]{

        background:white;

        border:2px solid #12411d;

        border-radius:24px;

        padding:10px;

        overflow:hidden;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }
                
        div[data-testid="stPlotlyChart"] > div {
    padding-bottom:0rem !important;
    margin-bottom:0rem !important;
}

.js-plotly-plot .plot-container {
    padding-bottom:0px !important;
    margin-bottom:0px !important;
}

svg.main-svg {
    border-radius:20px;
}

                /* METRIC CARD */
    .metric-box {
        background: white;
        border: 2px solid #12411d;
        border-radius: 20px;
        padding: 20px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: #1C6832;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .metric-icon svg {
        width: 26px;
        height: 26px;
        stroke: white;
        fill: none;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .metric-label {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-sub {
        color: #94A3B8;
        font-size: 11px;
        margin-top: 2px;
    }

    /* CHART CONTAINER */
    div[data-testid="stPlotlyChart"] {
        background: white;
        border: 2px solid #12411d;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-top: 10px;
    }

    /* DATAFRAME */
    div[data-testid="stDataFrame"] {
        background: white;
        border: 2px solid #12411d;
        border-radius: 20px;
        padding: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* SECTION TITLE */
    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #1E293B;
        margin-top: 24px;
        margin-bottom: 4px;
    }
    .section-sub {
        color: #64748B;
        font-size: 13px;
        margin-bottom: 10px;
    }

    /* TOOLBAR */
    .toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
                
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER
    # =========================
    st.title("Market Overview")

    st.markdown("""
    <div class="subtitle">
        Ringkasan kondisi pasar dan distribusi risiko ESG
    </div>
    """, unsafe_allow_html=True)

  # =========================
    # METRICS
    # =========================
    ICONS = {
        "stocks": '<svg viewBox="0 0 24 24"><rect x="2" y="3" width="8" height="8" rx="1"/><rect x="14" y="3" width="8" height="8" rx="1"/><rect x="2" y="13" width="8" height="8" rx="1"/><rect x="14" y="13" width="8" height="8" rx="1"/></svg>',
        "return": '<svg viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "risk":   '<svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "volume": '<svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    }

    c1, c2, c3, c4 = st.columns(4)

    for col, icon_key, label, value, sub in [
        (c1, "stocks",  "Total Stocks", str(len(features)),                          "Perusahaan"),
        (c2, "return",  "Avg Return",   str(round(features["return"].mean(), 4)),    "Rata-rata"),
        (c3, "risk",    "Avg Risk",     str(round(features["risk"].mean(), 4)),      "Rata-rata"),
        (c4, "volume",  "Avg Volume",   f"{int(df['Avg_Volume'].mean()):,}",         "Rata-rata"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-icon">{ICONS[icon_key]}</div>
                <div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CHARTS ROW
    # =========================
    left, right = st.columns([1.4, 1])

    # =========================
    # COLOR MAP
    # =========================
    COLOR_MAP = {
        "Moderate Risk": "#44B96F",
        "High Risk": "#F18B3B",
        "Severe Risk": "#6E78C9",
        "Low Risk": "#D77BB2",
        "Negligible Risk": "#94C973",
    }

    # =========================
    # CHARTS
    # =========================
    left, right = st.columns([1.6, 1])

    # =========================
    # BAR CHART
    # =========================
    with left:

        st.subheader("ESG Risk Distribution")
        st.caption(
            "Jumlah perusahaan berdasarkan kategori risiko ESG"
        )

        fig = px.bar(
            esg_counts,
            x="Kategori",
            y="Jumlah",
            color="Kategori",
            text="Jumlah",

            color_discrete_map=COLOR_MAP
        )

        fig.update_traces(
            marker_line_width=0,
            textposition="outside"
        )

        fig.update_layout(

        autosize=True,

        height=None,
        width=None,

        paper_bgcolor="white",
        plot_bgcolor="white",

        margin=dict(
            l=0,
            r=0,
            t=9,
            b=0
        ),


            font=dict(
                family="Poppins",
                size=13,
                color="#1E293B"
            ),

            xaxis=dict(
                title="Kategori Risiko",
                showgrid=False
            ),

            yaxis=dict(
                title="Jumlah",
                gridcolor="rgba(0,0,0,0.06)"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =========================
    # DONUT CHART
    # =========================
    with right:

        st.subheader(
            "Distribusi Persentase Risiko ESG"
        )

        st.caption(
            "Persentase total kategori risiko ESG"
        )

        pie = go.Figure(
            go.Pie(
                labels=esg_counts["Kategori"],
                values=esg_counts["Jumlah"],
                hole=0.68,

                marker=dict(
                    colors=[
                        COLOR_MAP.get(x, "#999")
                        for x in esg_counts["Kategori"]
                    ],
                    line=dict(
                        color="white",
                        width=2
                    )
                ),

                textinfo="percent",
            )
        )

        pie.update_layout(

            height=450,
                legend=dict(
                font=dict(size=11)
            ),

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),

            font=dict(
                family="Poppins",
                size=13,
                color="#1E293B"
            )
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    # =========================
    # TABLE
    # =========================
    st.subheader("Stock Ranking")

    st.caption(
        "Daftar saham berdasarkan score tertinggi"
    )

    ranking_df = (
        features
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index()[[
            "Stock_Name",
            "return",
            "risk",
            "score",
            "ESG_Category"
        ]]
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        height=500
    )