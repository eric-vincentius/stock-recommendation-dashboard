import streamlit as st
import pandas as pd

def show():
    st.markdown("""
    <h2 style='color:#FFFFFF;'>
        Historical Stock
    </h2>
    """, unsafe_allow_html=True)

    data = pd.read_csv("data/saham_data.csv")

    # Ensure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Select stock names (unique values)
    st.markdown("""
    <style>
    .custom-label {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-label">Select Stock</div>', unsafe_allow_html=True)

    stock = st.selectbox(
        label="",
        options=data["Stock_Name"].unique(),
        label_visibility="collapsed"
    )

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
    st.markdown(f"""
    <div style="font-size:18px; color:#FFFFFF;">
        ESG Risk:
        <span style="color:#FFFF00; font-weight:bold;">
            {df_stock['esg'].iloc[0]}
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <h2 style='color:#FFFF00;'>
        {stock} Closing Price
    </h2>
    """, unsafe_allow_html=True)
    st.line_chart(df_stock["Close"])