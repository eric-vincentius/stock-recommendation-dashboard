import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# PREPARE DATA
# =========================
def prepare_clustering_data(stock_summary):
    df = stock_summary.copy()

    df = df.rename(columns={
        "Avg_Return": "Mean_Return",
        "Return_Volatility": "Risk"
    })

    df = df[["Stock_Name", "Mean_Return", "Risk", "Avg_Volume"]]
    df = df.dropna()

    return df

# =========================
# SCALE DATA
# =========================
def scale_features(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Mean_Return", "Risk", "Avg_Volume"]])
    return X_scaled, scaler

# =========================
# ELBOW METHOD
# =========================
def compute_elbow(X_scaled, max_k=10):
    inertia = []

    for k in range(1, max_k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    return inertia

def get_cluster_profile(df, kmeans, scaler):
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    profile = pd.DataFrame(
        centers,
        columns=["Mean_Return", "Risk", "Avg_Volume"]
    )

    profile["Cluster"] = range(len(profile))
    return profile

def interpret_cluster(row):
    if row["Mean_Return"] > 0 and row["Risk"] < 1:
        return "Stable Growers"
    elif row["Mean_Return"] > 0 and row["Risk"] >= 1:
        return "High Growth High Risk"
    elif row["Mean_Return"] < 0 and row["Risk"] < 1:
        return "Weak Stable Stocks"
    else:
        return "Speculative / Declining"

# =========================
# RUN KMEANS
# =========================
def run_kmeans(df, X_scaled, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans