import streamlit as st
import pandas as pd

def show():
    st.title("📈 Historical Stock")

    data = pd.read_csv("data/saham_data.csv")

    # Ensure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Select stock names (unique values)
    stock = st.selectbox("Select Stock", data["Stock_Name"].unique())

    # Filter data
    df_stock = data[data["Stock_Name"] == stock].sort_values("Date")

    # Set index for proper time series chart
    df_stock = df_stock.set_index("Date")

    # Integrate with ESG data
    esg_data = pd.read_csv("data/esg_score.csv", sep=";")
    esg_data = esg_data.rename(columns={"Saham": "Stock_Name", "ESG Score": "esg"})
    esg_data["Stock_Name"] = esg_data["Stock_Name"].str.strip()
    df_stock["Stock_Name"] = df_stock["Stock_Name"].str.strip()
    df_stock = df_stock.merge(esg_data, on="Stock_Name", how="left")

    # Plot Close price
    st.write(f"ESG Risk: {df_stock['esg'].iloc[0]}")
    st.subheader(f"{stock} Closing Price")
    st.line_chart(df_stock["Close"])