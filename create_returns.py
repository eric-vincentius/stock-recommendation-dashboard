import pandas as pd

df = pd.read_csv("data/saham_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

# sort properly
df = df.sort_values(["Stock_Name", "Date"])

# compute returns per stock
df["Avg_Return"] = df.groupby("Stock_Name")["Close"].pct_change()

# save
df[["Date", "Stock_Name", "Avg_Return"]].to_csv(
    "data/stock_returns.csv",
    index=False
)

print("stock_returns.csv created!")