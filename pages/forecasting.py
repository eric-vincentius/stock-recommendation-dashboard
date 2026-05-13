import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models.forecasting import train_lstm, forecast_future


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/saham_data.csv")


# =========================
# PAGE
# =========================
def show():

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #355e2b,
            #5d824f
        );
    }

    .main-title {
        color: white;
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        font-family: Comic Sans MS;
    }

    .section-title {
        color: white;
        font-size: 30px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
        font-family: Comic Sans MS;
    }

    /* SELECTBOX */
    div[data-baseweb="select"] > div {

        background: #f4f4f4 !important;

        border: 4px solid #1e4b1d !important;

        border-radius: 18px !important;

        color: #222 !important;

        box-shadow:
            0 0 15px rgba(0,0,0,0.15),
            0 0 20px rgba(144,238,144,0.15);

    }

    /* BUTTON */
    .stButton > button {

        width: 100%;

        background: linear-gradient(
            135deg,
            #4d7f3f,
            #6ea15e
        );

        color: white;

        border: none;

        border-radius: 18px;

        padding: 14px;

        font-size: 18px;

        font-weight: bold;

        font-family: Comic Sans MS;

        box-shadow:
            0 0 15px rgba(0,0,0,0.15),
            0 0 20px rgba(144,238,144,0.20);

    }

    .stButton > button:hover {

        background: linear-gradient(
            135deg,
            #5c944a,
            #7db06b
        );

        color: white;

    }

    /* ALERTS */
    div[data-testid="stAlert"] {

        border-radius: 20px;

    }

    /* CHART */
    div[data-testid="stPlotlyChart"] {

        background: #f4f4f4;

        border-radius: 35px;

        padding: 10px;

        border: 6px solid #1e4b1d;

        overflow: hidden;

        box-shadow:
            0 0 25px rgba(0,0,0,0.2),
            0 0 35px rgba(144,238,144,0.25);

        margin-bottom: 25px;

    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown(
        '<div class="main-title"> Stock Price Forecast (LSTM)</div>',
        unsafe_allow_html=True
    )

    # =========================
    # LOAD DATA
    # =========================
    df = load_data()

    # =========================
    # DATE FORMAT
    # =========================
    df["Date"] = pd.to_datetime(df["Date"])

    # =========================
    # STOCK SELECT
    # =========================
    st.markdown(
        '<div class="section-title"> Select Stock</div>',
        unsafe_allow_html=True
    )

    stock_list = df["Stock_Name"].unique()

    selected_stock = st.selectbox(
        "",
        stock_list
    )

    # =========================
    # FILTER DATA
    # =========================
    df_stock = (
        df[df["Stock_Name"] == selected_stock]
        .sort_values("Date")
    )

    series = df_stock["Close"]

    # =========================
    # DATA CHECK
    # =========================
    if len(series) < 50:

        st.warning(" Not enough data for LSTM")

        return

    # =========================
    # FORECAST BUTTON
    # =========================
    if st.button(" Run Forecast"):

        with st.spinner("Training LSTM model..."):

            result = train_lstm(series)

        if result is None:

            st.error(
                " Model failed (not enough sequences)"
            )

            return

        # =========================
        # FUTURE FORECAST
        # =========================
        future = forecast_future(
            result["model"],
            result["last_seq"],
            result["scaler"],
            steps=120
        )

        st.success(" Forecast completed!")

        # =========================
        # HISTORICAL TITLE
        # =========================
        st.markdown(
            f'<div class="section-title">{selected_stock} Historical Prediction vs Actual</div>',
            unsafe_allow_html=True
        )

        # =========================
        # HISTORICAL DATA
        # =========================
        plot_df = pd.DataFrame({
            "Actual": result["y_test"],
            "Predicted": result["pred"]
        })

        # =========================
        # HISTORICAL CHART
        # =========================
        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                y=plot_df["Actual"],
                mode="lines",
                name="Actual",
                line=dict(
                    color="#4d7f3f",
                    width=4
                )
            )
        )

        fig1.add_trace(
            go.Scatter(
                y=plot_df["Predicted"],
                mode="lines",
                name="Predicted",
                line=dict(
                    color="#8bc34a",
                    width=4,
                    dash="dash"
                )
            )
        )

        fig1.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            height=600,

            font=dict(
                family="Comic Sans MS",
                size=16,
                color="#222"
            ),

            xaxis=dict(

                title="Time",

                title_font=dict(
                    size=22,
                    color="#1a1a1a"
                ),

                tickfont=dict(
                    size=14,
                    color="#333333"
                ),

                gridcolor="rgba(0,0,0,0.08)"
            ),

            yaxis=dict(

                title="Stock Price",

                title_font=dict(
                    size=22,
                    color="#1a1a1a"
                ),

                tickfont=dict(
                    size=14,
                    color="#333333"
                ),

                gridcolor="rgba(0,0,0,0.08)"
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)"
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # =========================
        # FUTURE TITLE
        # =========================
        st.markdown(
            f'<div class="section-title">🔮 {selected_stock} Future 120-Day Forecast</div>',
            unsafe_allow_html=True
        )

        # =========================
        # FUTURE DATA
        # =========================
        future_df = pd.DataFrame({
            "Forecast": future
        })

        # =========================
        # FUTURE CHART
        # =========================
        fig2 = px.line(
            future_df,
            y="Forecast"
        )

        fig2.update_traces(

            line=dict(
                color="#4d7f3f",
                width=5
            )

        )

        fig2.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            height=600,

            font=dict(
                family="Comic Sans MS",
                size=16,
                color="#222"
            ),

            xaxis=dict(

                title="Future Days",

                title_font=dict(
                    size=22,
                    color="#1a1a1a"
                ),

                tickfont=dict(
                    size=14,
                    color="#333333"
                ),

                gridcolor="rgba(0,0,0,0.08)"
            ),

            yaxis=dict(

                title="Forecast Price",

                title_font=dict(
                    size=22,
                    color="#1a1a1a"
                ),

                tickfont=dict(
                    size=14,
                    color="#333333"
                ),

                gridcolor="rgba(0,0,0,0.08)"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )