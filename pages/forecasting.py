import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models.forecasting import train_lstm, forecast_future


@st.cache_data
def load_data():
    return pd.read_csv("data/saham_data.csv")


def show():

    # =========================
    # CLEAN CSS
    # =========================
    st.markdown("""
    <style>

    .block-container {
        padding-top: 1.5rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 20px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="main-title">Stock Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">LSTM-based prediction of future stock prices</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    df = load_data()
    df["Date"] = pd.to_datetime(df["Date"])

    # =========================
    # SELECT + BUTTON ROW
    # =========================
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        stock_list = df["Stock_Name"].unique()
        selected_stock = st.selectbox("Select Stock", stock_list)

        st.markdown('</div>', unsafe_allow_html=True)

    # FILTER DATA
    df_stock = df[df["Stock_Name"] == selected_stock].sort_values("Date")
    series = df_stock["Close"]

    if len(series) < 50:

        st.warning(" Not enough data for LSTM")

        return

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        run = st.button("Run Forecast")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # RUN MODEL
    # =========================
    if run:

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

        # =========================
        # METRICS (KPI STYLE)
        # =========================
        mape = (abs(result["y_test"] - result["pred"]) / result["y_test"]).mean() * 100
        rmse = ((result["y_test"] - result["pred"]) ** 2).mean() ** 0.5

        col1, col2 = st.columns(2)

        col1.metric("MAPE (%)", f"{mape:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")

        # =========================
        # HISTORICAL PREDICTION
        # =========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Historical Prediction vs Actual")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=result["y_test"],
            mode="lines",
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            y=result["pred"],
            mode="lines",
            name="Predicted"
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # FUTURE DATA
        # =========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Future 120-Day Forecast")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            y=future,
            mode="lines",
            name="Forecast"
        ))

        fig2.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)