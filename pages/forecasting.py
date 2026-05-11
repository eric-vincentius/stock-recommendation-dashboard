import streamlit as st
import pandas as pd
from models.forecasting import train_lstm, forecast_future

@st.cache_data
def load_data():
    return pd.read_csv("data/saham_data.csv")

def show():
    st.title("📈 Stock Price Forecast (LSTM)")

    df = load_data()

    # Convert date properly
    df["Date"] = pd.to_datetime(df["Date"])

    # Stock selection
    stock_list = df["Stock_Name"].unique()
    selected_stock = st.selectbox("Select Stock", stock_list)

    # Filter data
    df_stock = df[df["Stock_Name"] == selected_stock].sort_values("Date")

    series = df_stock["Close"]

    # =========================
    # DATA CHECK
    # =========================
    if len(series) < 50:
        st.warning("Not enough data for LSTM")
        return

    # =========================
    # TRAIN / PREDICT
    # =========================
    if st.button("Run Forecast"):

        with st.spinner("Training LSTM model..."):
            result = train_lstm(series)

        if result is None:
            st.error("Model failed (not enough sequences)")
            return

        future = forecast_future(
            result["model"],
            result["last_seq"],
            result["scaler"],
            steps=120
        )

        st.success("Forecast completed!")

        # =========================
        # PLOT HISTORICAL PREDICTION
        # =========================
        st.subheader("Historical Prediction vs Actual")
        plot_df = pd.DataFrame({
            "Actual": result["y_test"],
            "Predicted": result["pred"]
        })

        st.line_chart(plot_df)

        # =========================
        # FUTURE FORECAST
        # =========================
        st.subheader("Future 120-Day Forecast")
        future_df = pd.DataFrame({
            "Forecast": future
        })

        st.line_chart(future_df)