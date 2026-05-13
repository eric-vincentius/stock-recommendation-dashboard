import streamlit as st
import pandas as pd
from models.forecasting import train_lstm, forecast_future

@st.cache_data
def load_data():
    return pd.read_csv("data/saham_data.csv")

def show():
    st.markdown("""
    <h2 style='color:#FFFFFF;'>
        Stock Price-Forecasting
    </h2>
    """, unsafe_allow_html=True)

    df = load_data()

    # Convert date properly
    df["Date"] = pd.to_datetime(df["Date"])

    # Stock selection
    st.markdown("""
    <style>
    .custom-label {
        color: #9AA0A6;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .custom-title {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-title">Select Stock</div>', unsafe_allow_html=True)

    # Selectbox (label hidden)
    stock_list = df["Stock_Name"].unique()
    selected_stock = st.selectbox(
        "",
        stock_list,
        label_visibility="collapsed"
    )

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
    st.markdown("""
        <style>
        [data-testid="stSpinner"] {
            color: #FFFFFF;
            font-size: 18px;
            font-weight: 600;
        }
                
        /* Success box */
        [data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
            background-color: #FFFFFF;
            color: #4CAF50;
            border-radius: 10px;
        }

        /* Error box */
        [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
            background-color: #FFFFFF;
            color: #F44336;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    
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
        st.markdown("""
            <h2 style='color:#FFFFFF;'>
                Historical Prediction vs Actual
            </h2>
            """, unsafe_allow_html=True)
        plot_df = pd.DataFrame({
            "Actual": result["y_test"],
            "Predicted": result["pred"]
        })

        st.line_chart(plot_df)

        # =========================
        # FUTURE FORECAST
        # =========================
        st.markdown("""
            <h2 style='color:#FFFFFF;'>
                Future 120-Day Forecast
            </h2>
            """, unsafe_allow_html=True)
        future_df = pd.DataFrame({
            "Forecast": future
        })

        st.line_chart(future_df)