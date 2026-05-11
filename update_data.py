from feature_engineering import process_stock_data

# after you load saham_data from yfinance

fact_stock_daily, dim_stock, fact_market_daily, fact_latest_snapshot = process_stock_data(saham_data)

# save
fact_latest_snapshot.to_csv("data/latest_snapshot.csv", index=False)
dim_stock.to_csv("data/stock_summary.csv", index=False)