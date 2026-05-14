import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score

from models.clustering import (
    prepare_clustering_data,
    scale_features,
    get_cluster_profile,
    interpret_cluster,
    run_kmeans
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Clustering",
    layout="wide"
)

# =========================
# MAIN PAGE
# =========================
def show():

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #F4F6F9;
    }

    /* MAIN TITLE */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0px;
    }

    /* SUBTITLE */
    .subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* METRIC CARD */
    .metric-card {
        background: white;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid #f3f4f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 90px;
    }

    .metric-left {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .metric-title {
        color: #6b7280;
        font-size: 13px;
        font-weight: 500;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }

    .metric-sub {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 2px;
    }

    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }

    .icon-green  { background: #d1fae5; }
    .icon-purple { background: #ede9fe; }
    .icon-yellow { background: #fef3c7; }

    /* INFO BOX */
    .info-box {
        background: #fafafa;
        border: 1px solid #eeeeee;
        padding: 18px 20px;
        border-radius: 14px;
        color: #4b5563;
        font-size: 14px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        height: 100%;
        box-sizing: border-box;
    }

    .info-icon {
        font-size: 20px;
        flex-shrink: 0;
        margin-top: 2px;
        color: #9ca3af;
    }

    .info-text {
        line-height: 1.6;
    }

    /* SECTION TITLE */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* TABLE */
    table {
        width: 100%;
        border-collapse: collapse;
    }

    th {
        background: #f9fafb !important;
        text-align: center !important;
        padding: 10px !important;
        font-size: 13px !important;
        font-family: 'Poppins', sans-serif !important;
    }

    td {
        text-align: center !important;
        padding: 10px !important;
        font-size: 13px !important;
        font-family: 'Poppins', sans-serif !important;
    }

    tbody tr:hover {
        background: #f9fafb;
    }

    /* LABEL BADGE */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
    }

    .badge-high   { background:#fee2e2; color:#b91c1c; }
    .badge-stable { background:#d1fae5; color:#065f46; }

    /* CARD WRAPPER */
    .card-wrap {
        background: white;
        border-radius: 18px;
        border: 1px solid #f3f4f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        padding: 20px 24px;
        margin-bottom: 0;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER
    # =========================
    st.markdown('<div class="main-title">Stock Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Segmentasi saham berdasarkan return, risk, dan volume</div>', unsafe_allow_html=True)

    # =========================
    # LOAD DATA
    # =========================
    @st.cache_data(ttl=3600)
    def load_data():
        saham_data = pd.read_csv("data/saham_data.csv")
        latest     = pd.read_csv("data/latest_snapshot.csv")
        summary    = pd.read_csv("data/stock_summary.csv")
        return saham_data, latest, summary

    saham_data, latest, summary = load_data()

    # =========================
    # PREPARE CLUSTERING
    # =========================
    df_cluster        = prepare_clustering_data(summary)
    X_scaled, scaler  = scale_features(df_cluster)
    df_cluster, kmeans = run_kmeans(df_cluster, X_scaled, k=3)

    # =========================
    # SILHOUETTE SCORE
    # =========================
    sil_score = silhouette_score(X_scaled, df_cluster["Cluster"])

    # =========================
    # CLUSTER PROFILE
    # =========================
    profile = get_cluster_profile(df_cluster, kmeans, scaler)

    # FIX: pastikan setiap cluster mendapat label unik
    # interpret_cluster dipanggil per baris; jika menghasilkan duplikat,
    # kita rename secara eksplisit berdasarkan urutan Mean_Return
    profile["Label"] = profile.apply(interpret_cluster, axis=1)

    # Deduplicate labels: jika ada label kembar, tambahkan suffix berdasarkan rank volume
    if profile["Label"].duplicated().any():
        seen   = {}
        labels = []
        for _, row in profile.iterrows():
            base = row["Label"]
            if base in seen:
                seen[base] += 1
                labels.append(f"{base} {seen[base]}")
            else:
                seen[base] = 1
                labels.append(base)
        profile["Label"] = labels

    df_cluster = df_cluster.merge(
        profile[["Cluster", "Label"]],
        on="Cluster",
        how="left"
    )

    # =========================
    # WARNA KONSISTEN
    # =========================
    unique_labels   = sorted(df_cluster["Label"].unique())
    palette         = px.colors.qualitative.Set2
    color_map       = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}

    # =========================
    # TOP SUMMARY ROW
    # =========================
    top1, top2, top3 = st.columns([3, 3, 3])

   
    with top1:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #12411d; border-width:2px">
            <div class="metric-left" ">
                <div class="metric-title">Total Stocks</div>
                <div class="metric-value">{len(df_cluster)}</div>
                <div class="metric-sub">Saham dianalisis</div>
            </div>
            <div class="metric-icon icon-green">📈</div>
        </div>
        """, unsafe_allow_html=True)

    with top2:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #12411d; border-width:2px">
            <div class="metric-left">
                <div class="metric-title">Total Clusters</div>
                <div class="metric-value">{df_cluster['Cluster'].nunique()}</div>
                <div class="metric-sub">Cluster terbentuk</div>
            </div>
            <div class="metric-icon icon-purple">👥</div>
        </div>
        """, unsafe_allow_html=True)

    with top3:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #12411d; border-width:2px" >
            <div class="metric-left">
                <div class="metric-title">Silhouette Score</div>
                <div class="metric-value">{sil_score:.2f}</div>
                <div class="metric-sub">Kualitas cluster</div>
            </div>
            <div class="metric-icon icon-yellow">⭐</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # SCATTER CHART
    # =========================
    fig1 = px.scatter(
        df_cluster,
        x="Mean_Return",
        y="Risk",
        color="Label",
        hover_name="Stock_Name",
        size="Avg_Volume",
        size_max=30,
        color_discrete_map=color_map,
        category_orders={"Label": unique_labels},   # FIX: urutan legend konsisten
    )

    fig1.update_traces(
        marker=dict(line=dict(width=0.5, color="#333"))
    )

    fig1.update_layout(
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Mean Return",
        yaxis_title="Risk",
        legend_title_text="Label",
    )

    
    st.markdown('<div class="section-title">Cluster Visualization</div>', unsafe_allow_html=True)
    st.caption("Hover untuk melihat nama saham")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<style>

.chart-card{

    background:white;

    border:2px solid #12411d;

    border-radius:24px;

    padding:20px;

    box-shadow:
        0 2px 10px rgba(0,0,0,0.04);

    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

    # =========================
    # DISTRIBUTION CHART  (FIX: gunakan color_discrete_map agar warna sama)
    # =========================
    cluster_counts = (
        df_cluster["Label"]
        .value_counts()
        .reset_index()
    )
    cluster_counts.columns = ["Cluster", "Count"]

    # Pastikan semua label muncul meski count-nya 0
    for lbl in unique_labels:
        if lbl not in cluster_counts["Cluster"].values:
            cluster_counts = pd.concat(
                [cluster_counts, pd.DataFrame({"Cluster": [lbl], "Count": [0]})],
                ignore_index=True
            )

    fig2 = px.bar(
        cluster_counts,
        x="Cluster",
        y="Count",
        text="Count",
        color="Cluster",
        color_discrete_map=color_map,    # FIX: warna sama dengan scatter
        category_orders={"Cluster": unique_labels},
    )

    fig2.update_traces(textposition="outside")

    fig2.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="",
        yaxis_title="Count",
    )

    # =========================
    # PROFILE TABLE
    # =========================
      # =========================
    # PROFILE TABLE
    # =========================

    def row_color(row):

        label = str(row["Label"]).lower()

        # HIGH RISK
        if "high" in label:

            return [
                "background-color:#FEF2F2; color:#111827;"
            ] * len(row)

        # MODERATE
        elif "moderate" in label:

            return [
                "background-color:#EFF6FF; color:#111827;"
            ] * len(row)

        # LOW
        elif "low" in label:

            return [
                "background-color:#F0FDF4; color:#111827;"
            ] * len(row)

        # DEFAULT
        else:

            return [
                "background-color:white; color:#111827;"
            ] * len(row)

    styled_profile = (

        profile.style

        .apply(
            row_color,
            axis=1
        )

        .format({
            "Mean_Return": "{:.6f}",
            "Risk": "{:.6f}",
            "Avg_Volume": "{:,.0f}",
        })

        .set_properties(**{
            "text-align": "center",
            "font-size": "13px",
            "border-color": "#12411d",
            "background-color": "white"
        })

        .set_table_styles([
            {
                "selector": "th",
                "props": [

                    ("background", "linear-gradient(90deg, #0B2E13 0%, #12411d 100%)"),
                    ("color", "black"),
                    ("font-weight", "700"),
                    ("border", "1px solid #12411d"),
                    ("padding", "12px"),
                    ("font-size", "13px"),
                    ("text-align", "center"),
                            ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #12411d"),
                    ("padding", "12px"),
                ]
            }
        ])
    )
        
   # =========================
    # ASSIGNMENT TABLE
    # =========================
    assignment_df = df_cluster[[
        "Stock_Name",
        "Mean_Return",
        "Risk",
        "Avg_Volume",
        "Label",
        "Cluster"
    ]].copy()

    assignment_df["Mean_Return"] = assignment_df["Mean_Return"].round(6)
    assignment_df["Risk"] = assignment_df["Risk"].round(6)

    # =========================
    # PAGINATION
    # =========================
    ROWS_PER_PAGE = 10

    total_rows = len(assignment_df)
    total_pages = (total_rows - 1) // ROWS_PER_PAGE + 1

    # SESSION STATE
    if "page" not in st.session_state:
        st.session_state.page = 1

    # =========================
    # CURRENT PAGE DATA
    # =========================
    start_idx = (st.session_state.page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE

    page_df = assignment_df.iloc[start_idx:end_idx]

    # =========================
    # STYLE TABLE
    # =========================
    styled_assignment = (
        page_df.style
        .background_gradient(subset=["Mean_Return"], cmap="Greens")
        .background_gradient(subset=["Risk"], cmap="Reds")
        .background_gradient(subset=["Avg_Volume"], cmap="Blues")
        .format({
            "Avg_Volume": "{:,.0f}"
        })
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px"
        })
    )

   

    st.markdown(
        '<div class="section-title">Stock Cluster Assignments</div>',
        unsafe_allow_html=True
    )

    st.write(
        styled_assignment.to_html(),
        unsafe_allow_html=True
    )

    # =========================
    # PAGINATION BELOW TABLE
    # =========================
    left, center, right = st.columns([1,2,1])

    with left:
        if st.session_state.page > 1:
            if st.button("⬅ Previous"):
                st.session_state.page -= 1
                st.rerun()

    with center:
        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:15px;
                font-weight:600;
                color:#12411d;
                padding-top:8px;
            ">
                Page {st.session_state.page} of {total_pages}
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        if st.session_state.page < total_pages:
            if st.button("Next ➡"):
                st.session_state.page += 1
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)