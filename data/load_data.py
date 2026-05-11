import yfinance as yf
import pandas as pd

tickers = [
    "MAPI.JK","ACES.JK","ADRO.JK","AKRA.JK","AMRT.JK","ASII.JK",
    "BBNI.JK","CPIN.JK","EXCL.JK","GGRM.JK","ICBP.JK","INCO.JK",
    "INDF.JK","INKP.JK","INTP.JK","ITMG.JK","KLBF.JK","MEDC.JK",
    "PGAS.JK","PTBA.JK","SMGR.JK","UNTR.JK","UNVR.JK","ANTM.JK",
    "BBCA.JK","BBRI.JK","BMRI.JK","BRPT.JK","TLKM.JK"
]

data = yf.download(tickers, start="2020-01-01")

df = data.stack(level=1).reset_index()

df.rename(columns={
    "Ticker": "Stock_Name"
}, inplace=True)

df["Stock_Name"] = df["Stock_Name"].str.replace(".JK", "", regex=False)
df = df.sort_values(by="Stock_Name", ascending=True)
df.to_csv("data/saham_data.csv", index=False)