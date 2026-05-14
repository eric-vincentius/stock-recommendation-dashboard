import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


def show():

    # ========================= 
    # CSS
    # =========================
    st.markdown("""
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
                html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp{
        background:#F4F6F9;
    }

    /* TITLE */
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* SEARCH INPUT OVERRIDE */
    [data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
    }

    /* SELECT BOX */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
    }

    /* DATE INPUT */
    [data-testid="stDateInput"] input {
        border-radius: 10px !important;
    }

    /* METRIC CARD */
    [data-testid="stMetric"] {
        background: white;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }

    [data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #6b7280 !important;
    }

    /* ESG CARD */
    .esg-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        position: relative;
        overflow: hidden;
        height: 100%;
    }

    .esg-label {
        font-size: 13px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .esg-value {
        font-size: 40px;
        font-weight: 800;
        color: #14532d;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    .esg-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #dcfce7;
        color: #166534;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
    }

    .esg-badge::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16a34a;
        display: inline-block;
    }

    .esg-leaf {
        position: absolute;
        right: 12px;
        bottom: 10px;
        font-size: 60px;
        opacity: 0.12;
    }

    /* CHART CARD */
   

    .chart-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }

    /* TIME FILTER BUTTONS */
    .stButton > button {
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 4px 12px !important;
        height: 34px !important;
        border: 1px solid #e5e7eb !important;
        background: white !important;
        color: #374151 !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background: #f0fdf4 !important;
        color: #15803d !important;
        border-color: #86efac !important;
    }

    /* ACTIVE TIME BUTTON (selected) */
    .active-btn > button {
        background: #14532d !important;
        color: white !important;
        border-color: #14532d !important;
    }

    /* METRICS ROW */
    .metric-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 20px;
    }

   .metric-item {
    flex: 1;
    min-width: 120px;
    background: white;

    border-radius: 16px;
    padding: 16px 18px;

    border: 2px solid #12411d;

    box-shadow: 0 2px 10px rgba(0,0,0,0.05);

    transition: all 0.2s ease;
}

.metric-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(18,65,29,0.12);
}

    .metric-item-label {
        font-size: 12px;
        color: #9ca3af;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .metric-item-value {
        font-size: 20px;
        font-weight: 800;
        color: #15803d;
        margin-bottom: 2px;
    }

    .metric-item-sub {
        font-size: 11px;
        color: #9ca3af;
    }

    .metric-item-value.positive {
        color: #15803d;
    }

    .metric-item-value.neutral {
        color: #1f2937;
    }

    /* SEARCH ROW */
    .search-row {
        background: white;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        border-color: #12411d;
        border-width: 2px;
        
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="main-title">Historical Stock</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze stock performance and ESG risk</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv("data/saham_data.csv")
    data["Date"] = pd.to_datetime(data["Date"])

    esg_data = pd.read_csv("data/esg_score.csv", sep=";")
    esg_data = esg_data.rename(columns={
        "Saham": "Stock_Name",
        "ESG Score": "esg"
    })

    all_stocks = sorted(data["Stock_Name"].unique().tolist())

    # =========================
    # SEARCH + DATE + SELECT ROW
    # =========================
    col_search, col_date, col_esg = st.columns([3, 2, 1.5])

    with col_search:
        search_query = st.text_input(
            label="search",
            placeholder=" Search stock...",
            label_visibility="collapsed"
        )

    with col_date:
        today = datetime.today().date()
        date_range = st.date_input(
            label="date_range",
            value=(today - timedelta(days=365), today),
            label_visibility="collapsed"
        )

    # STOCK SELECT — below search
    filtered_stocks = [s for s in all_stocks if search_query.lower() in s.lower()] if search_query else all_stocks
    st.markdown("""
    <style>
    .stock-select-wrapper {
        margin-top: 0px;
    }
    </style>

    <div class="stock-select-wrapper">
    """, unsafe_allow_html=True)

    col_select, _ = st.columns([3, 2.5])

    col_select, _ = st.columns([3, 2.5])
    with col_select:
        stock = st.selectbox(
            "Select Stock",
            filtered_stocks if filtered_stocks else all_stocks
        )

    # =========================
    # FILTER DATA
    # =========================
    df_stock = data[data["Stock_Name"] == stock].sort_values("Date")

    df_stock = df_stock.merge(esg_data, on="Stock_Name", how="left")

    esg_score = round(df_stock["esg"].iloc[0], 2) if not df_stock["esg"].isna().all() else 0.0

    # ESG badge label
    if esg_score < 20:
        risk_label = "Low Risk"
    elif esg_score < 40:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    # ESG CARD (top right) — using custom HTML
    with col_esg:
        st.markdown(f"""
        <div class="esg-card" style="border-color:#12411d; box-shadow: 0 4px 18px rgba(18,65,29,0.12); border-width: 3px;">
            <div class="esg-label">ESG Risk Score</div>
            <div class="esg-value">{esg_score}</div>
            <div class="esg-badge">{risk_label}</div>
            <div class="esg-leaf">🌿</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # TIME FILTER BUTTONS
    # =========================
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    chart_col, btn_col = st.columns([3, 2])

    with chart_col:
        st.markdown(f'<div class="chart-title">{stock} Price Trend</div>', unsafe_allow_html=True)

    time_filters = ["1M", "6M", "1Y", "3Y", "5Y", "Max"]

    if "time_filter" not in st.session_state:
        st.session_state.time_filter = "Max"

    with btn_col:
        btn_cols = st.columns(len(time_filters))
        for i, tf in enumerate(time_filters):
            with btn_cols[i]:
                if st.button(tf, key=f"tf_{tf}"):
                    st.session_state.time_filter = tf

    # Apply time filter
    selected_tf = st.session_state.time_filter
    today_dt = pd.Timestamp.today()

    tf_map = {
        "1M": today_dt - pd.DateOffset(months=1),
        "6M": today_dt - pd.DateOffset(months=6),
        "1Y": today_dt - pd.DateOffset(years=1),
        "3Y": today_dt - pd.DateOffset(years=3),
        "5Y": today_dt - pd.DateOffset(years=5),
        "Max": df_stock["Date"].min(),
    }

    start_date = tf_map.get(selected_tf, df_stock["Date"].min())
    df_plot = df_stock[df_stock["Date"] >= start_date]

    # =========================
    # CHART
    # =========================
   
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["Date"],
        y=df_plot["Close"],
        mode="lines",
        line=dict(color="#1a5c34", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(20,83,45,0.08)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#9ca3af"),
            title=dict(text="Date", font=dict(size=12, color="#9ca3af")),
            showline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f3f4f6",
            tickfont=dict(size=11, color="#9ca3af"),
            title=dict(text="Close", font=dict(size=12, color="#9ca3af")),
            showline=False,
        ),
        hovermode="x unified",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # METRICS ROW (Bottom)
    # =========================
    if not df_stock.empty:
        latest = df_stock.iloc[-1]
        prev = df_stock.iloc[-2] if len(df_stock) > 1 else latest

        current_price = latest["Close"]
        change_1d = current_price - prev["Close"]
        change_pct = (change_1d / prev["Close"]) * 100 if prev["Close"] != 0 else 0

        high_52w = df_stock[df_stock["Date"] >= today_dt - pd.DateOffset(years=1)]["Close"].max()
        low_52w = df_stock[df_stock["Date"] >= today_dt - pd.DateOffset(years=1)]["Close"].min()
        high_52w_date = df_stock.loc[df_stock["Close"] == high_52w, "Date"].iloc[0].strftime("%b %d, %Y") if not df_stock[df_stock["Close"] == high_52w].empty else ""
        low_52w_date = df_stock.loc[df_stock["Close"] == low_52w, "Date"].iloc[0].strftime("%b %d, %Y") if not df_stock[df_stock["Close"] == low_52w].empty else ""

        avg_vol = df_stock["Volume"].mean() if "Volume" in df_stock.columns else 0
       

        updated_str = latest["Date"].strftime("Updated: %b %d, %Y") if hasattr(latest["Date"], "strftime") else ""

        arrow = "↗" if change_1d >= 0 else "↘"
        change_color = "#15803d" if change_1d >= 0 else "#dc2626"
        sign = "+" if change_1d >= 0 else ""

        def fmt_large(v):
            if v >= 1e9:
                return f"{v/1e9:.2f}B"
            elif v >= 1e6:
                return f"{v/1e6:.2f}M"
            elif v >= 1e3:
                return f"{v/1e3:.2f}K"
            return f"{v:.2f}"

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-item-label">Current Price</div>
                <div class="metric-item-value neutral">{current_price:,.2f}</div>
                <div class="metric-item-sub">{updated_str}</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Change (1D)</div>
                <div class="metric-item-value" style="color:{change_color}">{sign}{change_1d:.2f} ({sign}{change_pct:.2f}%) {arrow}</div>
                <div class="metric-item-sub">&nbsp;</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">52W High</div>
                <div class="metric-item-value neutral">{high_52w:,.2f}</div>
                <div class="metric-item-sub">{high_52w_date}</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">52W Low</div>
                <div class="metric-item-value neutral">{low_52w:,.2f}</div>
                <div class="metric-item-sub">{low_52w_date}</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Avg. Volume</div>
                <div class="metric-item-value neutral">{fmt_large(avg_vol)}</div>
                <div class="metric-item-sub">Shares</div>
            </div>
           
        </div>
        """, unsafe_allow_html=True)