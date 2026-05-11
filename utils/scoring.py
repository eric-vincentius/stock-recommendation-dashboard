# utils/scoring.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_features(df, esg_scores):
    df = df.copy()

    df = df.rename(columns={
        "Avg_Return": "return",
        "Return_Volatility": "risk"
    })

    df = df[["return", "risk"]].dropna()

    df["esg"] = esg_scores

    scaler = MinMaxScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[["return", "risk", "esg"]]),
        columns=["return", "risk", "esg"],
        index=df.index
    )

    df["score"] = (
        0.4 * df_scaled["return"]
        - 0.2 * df_scaled["risk"]
        + 0.4 * df_scaled["esg"]
    )

    return df