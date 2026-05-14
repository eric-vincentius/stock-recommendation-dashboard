import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# COMPUTE GREEN SCORE
# =========================
def compute_green_score(df):

    df["NormESG"] = (
        100 - df["esg"]
    ) / 100

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

    df = df.drop(
        columns=[
            "NormESG",
            "NormReturn",
            "NormRisk"
        ]
    )

    return df


# =========================
# PAGE
# =========================
def show():

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
        padding-top:1.5rem;
        padding-left:1.5rem;
        padding-right:1.5rem;
        max-width:100%;
    }

    /* =========================
       TITLE
    ========================= */

    .main-title{
        color:#123524;
        font-size:38px;
        font-weight:800;
        margin-bottom:0;
    }

    .subtitle{
        color:#64748B;
        font-size:15px;
        margin-bottom:25px;
    }

    /* =========================
       CARD
    ========================= */

    .card{

        background:white;

        border:2px solid #12411d;

        border-radius:22px;

        padding:22px;

        box-shadow:
        0 2px 10px rgba(0,0,0,0.05);

        margin-bottom:22px;
    }

    /* =========================
       SECTION TITLE
    ========================= */

    .section-title{
        color:#123524;
        font-size:22px;
        font-weight:800;
        margin-bottom:4px;
    }

    .section-sub{
        color:#64748B;
        font-size:13px;
        margin-bottom:14px;
    }

    /* =========================
       PLOTLY CARD
    ========================= */

    div[data-testid="stPlotlyChart"]{

        background:white;

        border:2px solid #12411d;

        border-radius:22px;

        padding:18px;

        overflow:hidden;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    /* =========================
       TABLE
    ========================= */

    table{
        width:100% !important;
        border-collapse:collapse !important;
        overflow:hidden;
        border-radius:18px;
        border:2px solid #12411d;
    }

    thead th{

        background:#12411d !important;

        color:white !important;

        font-weight:700 !important;

        text-align:center !important;

        padding:12px !important;

        border:1px solid #d1d5db !important;
    }

    tbody td{

        text-align:center !important;

        padding:10px !important;

        border:1px solid #e5e7eb !important;

        color:#111827 !important;

        font-size:14px !important;
    }

    tbody th{

        text-align:center !important;

        padding:10px !important;

        border:1px solid #e5e7eb !important;

        background:white !important;

        color:#111827 !important;
    }

    tbody tr:nth-child(even){

        background:#F8FAFC !important;
    }

    tbody tr:hover{

        background:#DCFCE7 !important;
    }

    h3{
        color:#123524 !important;
        font-weight:800 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown(
        '<div class="main-title">Investment Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Top green investment opportunities based on ESG, return and risk</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(
        "data/stock_summary.csv"
    )

    esg_df = pd.read_csv(
        "data/esg_score.csv",
        sep=";"
    )

    esg_df = esg_df.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    df["Stock_Name"] = (
        df["Stock_Name"].str.strip()
    )

    esg_df["Stock_Name"] = (
        esg_df["Stock_Name"].str.strip()
    )

    # =========================
    # MERGE
    # =========================
    df = df.merge(
        esg_df,
        on="Stock_Name",
        how="left"
    )

    # =========================
    # COMPUTE SCORE
    # =========================
    df = compute_green_score(df)

    # =========================
    # TOP 10
    # =========================
    top_df = (
        df.sort_values(
            "Green Score",
            ascending=False
        )
        .head(10)
    )

    # =====================================================
    # CHART TITLE CARD
    # =====================================================
    st.markdown("""
   
        <div class="section-title">
            Top Performing Stocks
        </div>

       
   
    """, unsafe_allow_html=True)

    # =====================================================
    # BAR CHART
    # =====================================================
    fig = px.bar(
        top_df,
        x="Stock_Name",
        y="Green Score",
        text_auto=".2f",
        color="Green Score",

        color_continuous_scale=[
            [0.0, "#C8E6C9"],
            [0.4, "#66BB6A"],
            [0.7, "#2E7D32"],
            [1.0, "#12411d"]
        ]
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        textfont=dict(
            size=12,
            color="#334155"
        )
    )

    fig.update_layout(

        height=500,

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
            color="#1E293B"
        ),

        xaxis=dict(
            title="Stock",
            showgrid=False,
            linecolor="#E5E7EB"
        ),

        yaxis=dict(
            title="Green Score",
            gridcolor="#F1F5F9",
            linecolor="#E5E7EB"
        ),

        coloraxis_colorbar=dict(
            title="Score"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # TABLE TITLE
    # =====================================================
    st.markdown("""
    
        <div class="section-title">
            Full Ranking
        </div>

       
   
    """, unsafe_allow_html=True)

    # =====================================================
    # TABLE DATA
    # =====================================================
    ranking_df = (
        df.sort_values(
            "Green Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking_df = ranking_df[[
        "Stock_Name",
        "Avg_Return",
        "Return_Volatility",
        "Avg_Volume",
        "esg",
        "Green Score"
    ]]

    # =====================================================
    # INDEX DARI 1
    # =====================================================
    ranking_df.insert(
        0,
        "No",
        range(1, len(ranking_df) + 1)
    )

    ranking_df = ranking_df.round(4)

    # =====================================================
    # STYLED TABLE
    # =====================================================
    ranking_df = ranking_df.reset_index(drop=True)
    styled_df = (

        ranking_df.style
        .hide(axis="index")

        .format({
            "#": "{}",
            "Avg_Return": "{:.4f}",
            "Return_Volatility": "{:.4f}",
            "Avg_Volume": "{:,.0f}",
            "esg": "{:.6f}",
            "Green Score": "{:.4f}"
        })

        .background_gradient(
            subset=["Green Score"],
            cmap="Greens"
        )

        .background_gradient(
            subset=["Avg_Return"],
            cmap="Blues"
        )

        .background_gradient(
            subset=["Return_Volatility"],
            cmap="Oranges"
        )

        .map(
            lambda v:
            "color:white; font-weight:700;"
            if isinstance(v, (int, float)) and v >= 0.75
            else "color:#111827;",
            subset=["Green Score"]
        )

        .set_properties(**{
            "text-align": "center",
            "font-size": "14px",
            "padding": "10px"
        })

        .set_table_styles([

            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("border-collapse", "collapse")
                ]
            }

        ])
    )

    # =====================================================
    # SHOW TABLE
    # =====================================================
    # =====================================================
    # PAGINATION
    # =====================================================
    ROWS_PER_PAGE = 10

    total_rows = len(ranking_df)

    total_pages = (
        total_rows // ROWS_PER_PAGE
        + (total_rows % ROWS_PER_PAGE > 0)
    )

    # =========================
    # PAGE SELECTOR
    # =========================
    page = st.selectbox(
        "Page to view table data",
        range(1, total_pages + 1)
    )

    # =========================
    # SLICE DATA
    # =========================
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE

    page_df = ranking_df.iloc[start_idx:end_idx]

    # =========================
    # STYLE TABLE
    # =========================
    styled_df = (

        page_df.style
        .hide(axis="index")

        .format({
            "Avg_Return": "{:.4f}",
            "Return_Volatility": "{:.4f}",
            "Avg_Volume": "{:,.0f}",
            "esg": "{:.6f}",
            "Green Score": "{:.4f}"
        })

        .background_gradient(
            subset=["Green Score"],
            cmap="Greens"
        )

        .background_gradient(
            subset=["Avg_Return"],
            cmap="Blues"
        )

        .background_gradient(
            subset=["Return_Volatility"],
            cmap="Oranges"
        )

        .map(
            lambda v:
            "color:white; font-weight:700;"
            if isinstance(v, (int, float)) and v >= 0.75
            else "color:#111827;",
            subset=["Green Score"]
        )

        .set_properties(**{
            "text-align": "center",
            "font-size": "14px",
            "padding": "10px"
        })

    )

    # =========================
    # SHOW TABLE
    # =========================
    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )

    # =========================
    # PAGE INFO
    # =========================
    st.caption(
        f"Showing {start_idx + 1} - "
        f"{min(end_idx, total_rows)} "
        f"of {total_rows} rows"
    )