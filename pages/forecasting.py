import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models.forecasting import train_lstm, forecast_future

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    return pd.read_csv(
        "data/saham_data.csv"
    )

# =========================================================
# PAGE
# =========================================================
def show():

    # =========================================================
    # CSS
    # =========================================================
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]{
        font-family:'Poppins', sans-serif;
    }

    .stApp{
        background:#F4F6F9;
    }

    .block-container{
        padding-top:1.2rem;
        padding-left:1.5rem;
        padding-right:1.5rem;
        max-width:100%;
    }

    /* =====================================================
       TITLE
    ===================================================== */

    .main-title{
        font-size:38px;
        font-weight:800;
        color:#123524;
        margin-bottom:0;
    }

    .subtitle{
        color:#64748B;
        font-size:15px;
        margin-bottom:20px;
    }

    /* =====================================================
       CARD
    ===================================================== */

    .card{

        background:white;

        border:2px solid #12411d;

        border-radius:20px;

        padding:18px;

        box-shadow:
        0 4px 12px rgba(0,0,0,0.04);

        margin-bottom:18px;
    }

    /* =====================================================
       METRIC CARD
    ===================================================== */

    .metric-card{

        background:white;

        border:2px solid #12411d;

        border-radius:18px;

        padding:18px;

        height:110px;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-title{

        color:#123524;

        font-size:15px;

        font-weight:700;

        margin-bottom:8px;
    }

    .metric-value{

        color:#12411d;

        font-size:48px;

        font-weight:800;
    }

    /* =====================================================
       SELECTBOX
    ===================================================== */

    div[data-baseweb="select"]{

        border:2px solid #D1D5DB;

        border-radius:12px;

        overflow:hidden;
    }

    div[data-baseweb="select"] > div{

        border:none !important;
    }

    /* =====================================================
       BUTTON
    ===================================================== */

    div.stButton > button{

        width:100%;

        background:#12411d;

        color:white;

        border:none;

        height:52px;

        border-radius:12px;

        font-weight:700;

        font-size:16px;

        transition:0.2s;
    }

    div.stButton > button:hover{

        background:#0E3417;

        color:white;
    }

    /* =====================================================
       PLOTLY CHART
    ===================================================== */

    div[data-testid="stPlotlyChart"]{

        background:white;

        border:2px solid #12411d;

        border-radius:20px;

        padding:16px;

        overflow:hidden;

        box-shadow:
        0 2px 8px rgba(0,0,0,0.04);

        margin-bottom:18px;
    }

    /* =====================================================
       HEADER
    ===================================================== */

    h3{
        color:#123524 !important;
        font-weight:800 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # TITLE
    # =========================================================
    st.markdown(
        '<div class="main-title">Stock Price Forecasting</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">LSTM-based prediction of future stock prices</div>',
        unsafe_allow_html=True
    )

    # =========================================================
    # LOAD DATA
    # =========================================================
    df = load_data()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # =========================================================
    # FILTER CARD
    # =========================================================
         # =========================================================
    # FILTER SECTION
    # =========================================================

    st.markdown("""
    <style>

    /* =====================================================
    FILTER CONTAINER
    ===================================================== */

    .filter-card{

        background:white;

        border:2px solid #12411d;

        border-radius:22px;

        padding:22px;

        margin-bottom:22px;

        box-shadow:
            0 4px 14px rgba(0,0,0,0.05);
    }

    /* =====================================================
    LABEL
    ===================================================== */

    .stSelectbox label,
    .stDateInput label{

        color:#123524 !important;

        font-weight:700 !important;

        font-size:14px !important;

        margin-bottom:6px !important;
    }

    /* =====================================================
    SELECTBOX
    ===================================================== */

    div[data-baseweb="select"]{

        border:2px solid #12411d !important;

        border-radius:16px !important;

        background:white !important;

        min-height:54px !important;

        overflow:hidden !important;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.05);

        transition:0.25s ease;
    }

    div[data-baseweb="select"] > div{

        border:none !important;

        background:transparent !important;
    }

    /* =====================================================
    DATE INPUT
    ===================================================== */

    div[data-baseweb="input"]{

        border:2px solid #12411d !important;

        border-radius:16px !important;

        background:white !important;

        min-height:54px !important;

        overflow:hidden !important;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.05);

        transition:0.25s ease;
    }

    div[data-baseweb="input"] > div{

        border:none !important;

        background:transparent !important;
    }

    /* =====================================================
    HOVER EFFECT
    ===================================================== */

    div[data-baseweb="select"]:hover,
    div[data-baseweb="input"]:hover{

        border-color:#1E6B34 !important;

        box-shadow:
            0 6px 16px rgba(18,65,29,0.15);
    }

    /* =====================================================
    BUTTON
    ===================================================== */

    .stButton > button{

        width:100%;

        height:54px;

        margin-top:30px;

        border:none;

        border-radius:16px;

        background:linear-gradient(
            135deg,
            #12411d,
            #1d6b2f
        );

        color:white !important;

        font-size:15px;

        font-weight:700;

        transition:0.25s ease;

        box-shadow:
            0 5px 16px rgba(18,65,29,0.25);
    }

    .stButton > button:hover{

        transform:translateY(-2px);

        box-shadow:
            0 8px 22px rgba(18,65,29,0.35);
    }

    /* =====================================================
    REMOVE EXTRA SPACE
    ===================================================== */

    [data-testid="column"]{

        padding-top:0rem !important;
    }

    </style>
    """, unsafe_allow_html=True)

   
    # =========================================================
    # COLUMNS
    # =========================================================

    col1, col2, col3 = st.columns(
        [3, 2.2, 1],
        gap="medium"
    )

    # =========================================================
    # STOCK SELECT
    # =========================================================

    with col1:

        stock_list = sorted(
            df["Stock_Name"].unique()
        )

        selected_stock = st.selectbox(
            "Select Stock",
            stock_list
        )

    # =========================================================
    # DATE INPUT
    # =========================================================

    with col2:

        selected_date = st.date_input(
            "Select Date (Optional)",
            value=None
        )

    # =========================================================
    # BUTTON
    # =========================================================

    with col3:

        run = st.button(
            "▶ Run Forecast",
            use_container_width=True
        )

    # =========================================================
    # CLOSE FILTER CARD
    # =========================================================

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # =========================================================
    # FILTER DATA
    # =========================================================
    df_stock = (
        df[df["Stock_Name"] == selected_stock]
        .sort_values("Date")
    )

    series = df_stock["Close"]

    if len(series) < 50:

        st.warning(
            "Not enough data for LSTM"
        )

        return

    # =========================================================
    # RUN MODEL
    # =========================================================
    if run:

        with st.spinner(
            "Training LSTM model..."
        ):

            result = train_lstm(series)

        if result is None:

            st.error(
                "Model failed (not enough sequences)"
            )

            return

        # =====================================================
        # FUTURE FORECAST
        # =====================================================
        future = forecast_future(
            result["model"],
            result["last_seq"],
            result["scaler"],
            steps=120
        )

        # =====================================================
        # METRICS
        # =====================================================
        mape = (
            abs(
                result["y_test"] -
                result["pred"]
            ) / result["y_test"]
        ).mean() * 100

        rmse = (
            (
                result["y_test"] -
                result["pred"]
            ) ** 2
        ).mean() ** 0.5

        
        col1, col2 = st.columns(2)

       

            

        # =====================================================
        # HISTORICAL CHART
        # =====================================================
        st.subheader(
            "Historical Prediction vs Actual"
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=result["y_test"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#1D4ED8",
                width=3
            )
        ))

        fig.add_trace(go.Scatter(
            y=result["pred"],
            mode="lines",
            name="Predicted",
            line=dict(
                color="#C026D3",
                width=3
            )
        ))

        fig.update_layout(

            height=420,

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            font=dict(
                family="Poppins",
                color="#123524"
            ),

            xaxis=dict(
                showgrid=False,
                title=""
            ),

            yaxis=dict(
                title="Price",
                gridcolor="#E5E7EB"
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================================
        # FUTURE FORECAST CHART
        # =====================================================
        st.subheader(
            "Future 120-Day Forecast"
        )

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            y=future,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#C026D3",
                width=4
            )
        ))

        fig2.update_layout(

            height=420,

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            font=dict(
                family="Poppins",
                color="#123524"
            ),

            xaxis=dict(
                title="Days Ahead",
                gridcolor="#F1F5F9"
            ),

            yaxis=dict(
                title="Price",
                gridcolor="#E5E7EB"
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )