import pandas as pd
import numpy as np

def process_stock_data(saham_data):

    g = saham_data.groupby("Stock_Name")

    # =========================
    # RETURNS
    # =========================
    saham_data["Prev_Close"] = g["Close"].shift(1)
    saham_data["Return"] = saham_data["Close"] / saham_data["Prev_Close"] - 1
    saham_data["Intraday_Return"] = saham_data["Close"] / saham_data["Open"] - 1
    saham_data["Gap_Return"] = saham_data["Open"] / saham_data["Prev_Close"] - 1
    saham_data["Range_Pct"] = (saham_data["High"] - saham_data["Low"]) / saham_data["Close"]

    saham_data["Return_5D"] = g["Close"].pct_change(5)
    saham_data["Return_20D"] = g["Close"].pct_change(20)

    # =========================
    # RISK FEATURES
    # =========================
    saham_data["Volatility20"] = g["Return"].rolling(20).std().reset_index(level=0, drop=True)
    saham_data["AvgVolume20"] = g["Volume"].rolling(20).mean().reset_index(level=0, drop=True)
    saham_data["VolumeRatio"] = saham_data["Volume"] / saham_data["AvgVolume20"]

    saham_data["Peak20"] = g["Close"].rolling(20).max().reset_index(level=0, drop=True)
    saham_data["Peak60"] = g["Close"].rolling(60).max().reset_index(level=0, drop=True)
    saham_data["Drawdown20"] = saham_data["Close"] / saham_data["Peak20"] - 1
    saham_data["Drawdown60"] = saham_data["Close"] / saham_data["Peak60"] - 1

    # =========================
    # ATR
    # =========================
    tr1 = saham_data["High"] - saham_data["Low"]
    tr2 = (saham_data["High"] - saham_data["Prev_Close"]).abs()
    tr3 = (saham_data["Low"] - saham_data["Prev_Close"]).abs()
    saham_data["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    saham_data["ATR14"] = g["TR"].rolling(14).mean().reset_index(level=0, drop=True)
    saham_data["ATR_Pct"] = saham_data["ATR14"] / saham_data["Close"]

    # =========================
    # CLEAN
    # =========================
    saham_data = saham_data.reset_index(drop=True)
    saham_data["Date"] = pd.to_datetime(saham_data["Date"])

    # =========================
    # STOCK SUMMARY (DIM)
    # =========================
    stock_summary = saham_data.groupby("Stock_Name").agg(
        Avg_Return=("Return", "mean"),
        Return_Volatility=("Return", "std"),
        Avg_ATR_Pct=("ATR_Pct", "mean"),
        Worst_Drawdown60=("Drawdown60", "min"),
        Avg_Volume=("Volume", "mean")
    ).reset_index()

    # =========================
    # MARKET DAILY
    # =========================
    market_daily = saham_data.groupby("Date").agg(
        Mean_Return=("Return", "mean"),
        Avg_Volatility20=("Volatility20", "mean")
    ).reset_index()

    # =========================
    # LATEST SNAPSHOT (IMPORTANT)
    # =========================
    latest_snapshot = saham_data.groupby("Stock_Name").tail(1).copy()

    latest_snapshot["VolRank"] = latest_snapshot["Volatility20"].rank(pct=True)
    latest_snapshot["ATRRank"] = latest_snapshot["ATR_Pct"].rank(pct=True)
    latest_snapshot["MoveRank"] = latest_snapshot["Return"].abs().rank(pct=True)
    latest_snapshot["DrawdownRank"] = (-latest_snapshot["Drawdown60"]).rank(pct=True)
    latest_snapshot["LiquidityRiskRank"] = (
        1 / latest_snapshot["AvgVolume20"].replace(0, np.nan)
    ).rank(pct=True)

    latest_snapshot["RiskScore"] = (
        0.30 * latest_snapshot["VolRank"] +
        0.20 * latest_snapshot["ATRRank"] +
        0.15 * latest_snapshot["MoveRank"] +
        0.15 * latest_snapshot["DrawdownRank"] +
        0.20 * latest_snapshot["LiquidityRiskRank"]
    ) * 100

    return saham_data, stock_summary, market_daily, latest_snapshot

saham_data, stock_summary, market_daily, latest_snapshot = process_stock_data(pd.read_csv("data/saham_data.csv"))
stock_summary.to_csv("data/stock_summary.csv", index=False)
market_daily.to_csv("data/market_daily.csv", index=False)
latest_snapshot.to_csv("data/latest_snapshot.csv", index=False)