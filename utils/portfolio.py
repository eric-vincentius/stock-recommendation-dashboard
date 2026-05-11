# utils/portfolio.py

import numpy as np
import pandas as pd

def portfolio_metrics(weights, returns_df):
    """
    weights: numpy array (n_assets,)
    returns_df: DataFrame of asset returns (time x assets)
    """

    # Clean data
    returns_df = returns_df.dropna()

    # Annualized mean returns
    mean_returns = returns_df.mean() * 252

    # Covariance matrix (annualized)
    cov_matrix = returns_df.cov() * 252

    # Portfolio return
    port_return = np.dot(weights, mean_returns)

    # Portfolio volatility (risk)
    port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

    return port_return, port_vol