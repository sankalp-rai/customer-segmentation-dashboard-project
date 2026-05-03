from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/customer_data.csv")
RANDOM_STATE = 42
CLUSTER_FEATURES = [
    "Age",
    "Annual Income",
    "Spending Score",
    "Purchase Frequency",
    "Total Spend",
    "Loyalty Score",
    "Customer Tenure",
    "Days Since Last Purchase",
]
COLOR_SEQUENCE = ["#1d4ed8", "#0f766e", "#f59e0b", "#be123c", "#6d28d9", "#0369a1", "#c2410c", "#475569"]


st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Apply a premium executive-dashboard visual system."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #eef3f9;
            --panel: #ffffff;
            --ink: #0f172a;
            --ink-soft: #1e293b;
            --muted: #64748b;
            --line: #d8e2ef;
            --accent: #1d4ed8;
            --accent-2: #0f766e;
            --warning: #b45309;
            --danger: #be123c;
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 32%, #f6f8fb 100%);
            color: var(--ink);
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 55%, #172033 100%);
            border-right: 1px solid rgba(226, 232, 240, 0.08);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #e2e8f0;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] input {
            background-color: #f8fafc;
            color: #0f172a;
            border-color: rgba(148, 163, 184, 0.5);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #0f172a;
        }

        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
            background-color: #1d4ed8;
            color: #ffffff;
            border-radius: 6px;
            font-weight: 650;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
            color: #e2e8f0;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1540px;
        }

        .executive-header {
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(29, 78, 216, 0.93) 52%, rgba(15, 118, 110, 0.92) 100%);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 10px;
            padding: 1.35rem 1.55rem;
            margin-bottom: 1.1rem;
            box-shadow: var(--shadow);
            color: #ffffff;
        }

        .brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.7rem;
        }

        .brand-mark {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            color: #dbeafe;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }

        .brand-pill {
            color: #ecfeff;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .brand-dot {
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            background: #38bdf8;
            box-shadow: 0 0 0 5px rgba(56, 189, 248, 0.16);
        }

        .dashboard-title {
            font-size: 2.15rem;
            font-weight: 850;
            letter-spacing: 0;
            color: #ffffff;
            margin: 0;
            line-height: 1.08;
        }

        .dashboard-subtitle {
            color: #dbeafe;
            margin-top: 0.45rem;
            max-width: 840px;
            font-size: 1rem;
            line-height: 1.45;
        }

        .section-title {
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 820;
            margin-top: 1rem;
            margin-bottom: 0.45rem;
            letter-spacing: 0;
        }

        .section-kicker {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: -0.3rem;
            margin-bottom: 0.65rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }

        .kpi-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1.05rem 1.1rem 1rem;
            min-height: 136px;
            box-shadow: var(--shadow);
        }

        .kpi-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: var(--kpi-color, var(--accent));
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 820;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.42rem;
        }

        .kpi-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 850;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }

        .kpi-note {
            color: var(--ink-soft);
            font-size: 0.9rem;
            line-height: 1.3;
        }

        .kpi-trend {
            display: inline-block;
            margin-top: 0.65rem;
            padding: 0.28rem 0.52rem;
            background: #eef6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 750;
        }

        .insight-panel {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid var(--line);
            border-top: 4px solid var(--card-color, var(--accent));
            border-radius: 10px;
            padding: 1rem 1.05rem;
            min-height: 142px;
            box-shadow: var(--shadow);
        }

        .insight-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }

        .insight-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .insight-copy {
            color: var(--ink-soft);
            font-size: 0.9rem;
            line-height: 1.35;
        }

        .recommendation-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow);
            min-height: 164px;
        }

        .recommendation-title {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 820;
            margin-bottom: 0.35rem;
        }

        .recommendation-copy {
            color: var(--ink-soft);
            font-size: 0.91rem;
            line-height: 1.42;
        }

        .recommendation-priority {
            display: inline-block;
            color: #ffffff;
            background: var(--priority-color, var(--accent));
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 780;
            margin-bottom: 0.65rem;
        }

        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.35rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.055);
        }

        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.045);
        }

        .stDownloadButton button {
            background: #0f172a;
            color: #ffffff;
            border: 1px solid #0f172a;
            border-radius: 8px;
            font-weight: 760;
            min-height: 2.7rem;
        }

        .stDownloadButton button:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
            color: #ffffff;
        }

        @media (max-width: 900px) {
            .dashboard-title {
                font-size: 1.55rem;
            }
            .brand-row {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.5rem;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .kpi-card,
            .insight-panel,
            .recommendation-panel {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading and cleaning customer data...")
def load_customer_data(path: Path) -> pd.DataFrame:
    """Load only needed columns with explicit dtypes for predictable large-CSV performance."""
    if not path.exists():
        st.error(f"Dataset not found at {path}.")
        st.stop()

    dtype_map = {
        "Customer ID": "string",
        "Age": "float32",
        "Gender": "category",
        "Annual Income": "float32",
        "Spending Score": "float32",
        "Purchase Frequency": "float32",
        "Region": "category",
        "Country": "category",
        "Preferred Category": "category",
        "Total Spend": "float32",
        "Loyalty Score": "float32",
        "Customer Tenure": "float32",
    }
    df = pd.read_csv(path, dtype=dtype_map, parse_dates=["Last Purchase Date"])
    return clean_customer_data(df)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names, impute missing values, and add analytics features."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    categorical_cols = ["Gender", "Region", "Country", "Preferred Category"]
    numeric_cols = [
        "Age",
        "Annual Income",
        "Spending Score",
        "Purchase Frequency",
        "Total Spend",
        "Loyalty Score",
        "Customer Tenure",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype("string").str.strip()
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        df[col] = df[col].replace("", "Unknown").astype("category")

    df["Last Purchase Date"] = pd.to_datetime(df["Last Purchase Date"], errors="coerce")
    fallback_date = df["Last Purchase Date"].median()
    df["Last Purchase Date"] = df["Last Purchase Date"].fillna(fallback_date)
    reference_date = df["Last Purchase Date"].max()
    df["Days Since Last Purchase"] = (reference_date - df["Last Purchase Date"]).dt.days.astype("float32")

    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 24, 34, 44, 54, 64, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    df["Customer Value"] = np.select(
        [
            (df["Total Spend"] >= df["Total Spend"].quantile(0.85)) & (df["Loyalty Score"] >= 70),
            (df["Days Since Last Purchase"] >= df["Days Since Last Purchase"].quantile(0.75)) & (df["Loyalty Score"] <= 45),
            (df["Annual Income"] >= df["Annual Income"].quantile(0.80)) & (df["Spending Score"] <= 55),
        ],
        ["High Value", "At Risk", "Premium Opportunity"],
        default="Core",
    )
    return df


@st.cache_data(show_spinner="Training segmentation model...")
def cluster_customers(df: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run K-Means on behavior and value features, then return labeled customers and segment profiles."""
    model_df = df.copy()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)),
        ]
    )
    model_df["Cluster"] = pipeline.fit_predict(model_df[CLUSTER_FEATURES]).astype(int)

    profiles = (
        model_df.groupby("Cluster", observed=True)
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual Income", "mean"),
            Avg_Spend=("Total Spend", "mean"),
            Avg_Spending_Score=("Spending Score", "mean"),
            Avg_Loyalty=("Loyalty Score", "mean"),
            Avg_Frequency=("Purchase Frequency", "mean"),
            Avg_Tenure=("Customer Tenure", "mean"),
            Avg_Recency=("Days Since Last Purchase", "mean"),
        )
        .reset_index()
    )
    profiles["Share"] = profiles["Customers"] / len(model_df)
    return model_df, profiles, float(pipeline.named_steps["kmeans"].inertia_)


@st.cache_data(show_spinner="Calculating elbow curve...")
def compute_elbow(df: pd.DataFrame, max_k: int = 10, sample_size: int = 25000) -> pd.DataFrame:
    """Use a deterministic sample so the elbow chart stays quick on large datasets."""
    sample = df[CLUSTER_FEATURES].sample(min(sample_size, len(df)), random_state=RANDOM_STATE)
    scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit_transform(sample)
    results = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled)
        results.append({"Clusters": k, "Inertia": model.inertia_})
    return pd.DataFrame(results)


@st.cache_data(show_spinner="Projecting clusters with PCA...")
def compute_pca(df: pd.DataFrame, sample_size: int = 12000) -> pd.DataFrame:
    """Project a sample to 2D for responsive browser rendering."""
    sample = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE).copy()
    scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit_transform(sample[CLUSTER_FEATURES])
    components = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    sample["PC1"] = components[:, 0]
    sample["PC2"] = components[:, 1]
    sample["Cluster"] = sample["Cluster"].astype(str)
    return sample


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar controls and return the filtered customer table."""
    st.sidebar.title("Segmentation Controls")
    st.sidebar.caption("Filter the labeled customer base for executive analysis.")
    n_clusters = st.sidebar.slider("K-Means clusters", min_value=3, max_value=8, value=5, step=1)

    region = st.sidebar.multiselect("Region", sorted(df["Region"].astype(str).unique()))
    country_options = sorted(df.loc[df["Region"].astype(str).isin(region) if region else df.index, "Country"].astype(str).unique())
    country = st.sidebar.multiselect("Country", country_options)
    gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].astype(str).unique()))
    category = st.sidebar.multiselect("Preferred Category", sorted(df["Preferred Category"].astype(str).unique()))
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

    filtered = df.copy()
    if region:
        filtered = filtered[filtered["Region"].astype(str).isin(region)]
    if country:
        filtered = filtered[filtered["Country"].astype(str).isin(country)]
    if gender:
        filtered = filtered[filtered["Gender"].astype(str).isin(gender)]
    if category:
        filtered = filtered[filtered["Preferred Category"].astype(str).isin(category)]
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]

    st.sidebar.divider()
    st.sidebar.metric("Filtered Customers", f"{len(filtered):,}")
    return filtered, n_clusters


def plot_template(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=54, b=34),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family='Inter, "Segoe UI", Arial, sans-serif', color="#0f172a", size=13),
        title=dict(font=dict(size=18, color="#0f172a"), x=0.02, xanchor="left"),
        legend_title_text="",
        colorway=COLOR_SEQUENCE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5edf7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5edf7", zeroline=False)
    return fig


def render_header() -> None:
    st.markdown(
        """
        <div class="executive-header">
            <div class="brand-row">
                <div class="brand-mark"><span class="brand-dot"></span>Customer Intelligence Suite</div>
                <div class="brand-pill">Executive segmentation cockpit</div>
            </div>
            <h1 class="dashboard-title">Customer Segmentation Dashboard</h1>
            <div class="dashboard-subtitle">
                Board-ready view of customer value, loyalty, purchase behavior, and machine-learning segments for commercial decision making.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str, trend: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="--kpi-color: {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
            <div class="kpi-trend">{trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame, cluster_count: int) -> None:
    total_customers = len(df)
    avg_spend = df["Total Spend"].mean() if total_customers else 0
    avg_loyalty = df["Loyalty Score"].mean() if total_customers else 0
    avg_frequency = df["Purchase Frequency"].mean() if total_customers else 0
    high_value_share = (df["Customer Value"].eq("High Value").mean() * 100) if total_customers else 0

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Total Customers", f"{total_customers:,}", "Filtered customer universe", f"{high_value_share:.1f}% high value", "#1d4ed8")
    with cols[1]:
        kpi_card("Average Spend", f"${avg_spend:,.0f}", "Mean lifetime commercial value", f"{avg_frequency:.1f} avg purchases", "#0f766e")
    with cols[2]:
        kpi_card("Average Loyalty", f"{avg_loyalty:.1f}", "Composite retention strength", "Score out of 100", "#7c3aed")
    with cols[3]:
        kpi_card("Active Segments", f"{cluster_count}", "K-Means portfolio groups", "Model-driven clusters", "#f59e0b")


def insight_card(label: str, value: str, copy: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="insight-panel" style="--card-color: {color};">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_business_insights(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No customers match the current filters.")
        return

    high_value = df[(df["Total Spend"] >= df["Total Spend"].quantile(0.85)) & (df["Loyalty Score"] >= 70)]
    at_risk = df[(df["Days Since Last Purchase"] >= df["Days Since Last Purchase"].quantile(0.75)) & (df["Loyalty Score"] <= 45)]
    premium = df[(df["Annual Income"] >= df["Annual Income"].quantile(0.80)) & (df["Spending Score"] <= 55)]

    st.markdown('<div class="section-title">Executive Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Commercial opportunity and risk signals recalculated from the current filter context.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        insight_card(
            "High-value customers",
            f"{len(high_value):,}",
            f"Average spend is ${high_value['Total Spend'].mean():,.0f}. Prioritize loyalty benefits and concierge retention.",
            "#0f766e",
        )
    with cols[1]:
        insight_card(
            "At-risk customers",
            f"{len(at_risk):,}",
            f"Average recency is {at_risk['Days Since Last Purchase'].mean():.0f} days. Trigger win-back offers and service follow-up.",
            "#be123c",
        )
    with cols[2]:
        insight_card(
            "Premium opportunities",
            f"{len(premium):,}",
            f"High income with moderate spend. Use premium bundles, category education, and tailored recommendations.",
            "#7c3aed",
        )


def recommendation_card(priority: str, title: str, copy: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="recommendation-panel">
            <div class="recommendation-priority" style="--priority-color: {color};">{priority}</div>
            <div class="recommendation-title">{title}</div>
            <div class="recommendation-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_strategic_recommendations(df: pd.DataFrame) -> None:
    if df.empty:
        return

    top_region = df.groupby("Region", observed=True)["Total Spend"].mean().idxmax()
    top_category = df.groupby("Preferred Category", observed=True)["Total Spend"].sum().idxmax()
    at_risk_share = df["Customer Value"].eq("At Risk").mean() * 100
    loyalty_gap = max(0, 75 - df["Loyalty Score"].mean())

    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Actionable next steps for revenue growth, retention, and premium conversion.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        recommendation_card(
            "Growth",
            f"Scale premium plays in {top_region}",
            f"{top_region} has the strongest average spend in this view. Allocate premium offers, relationship campaigns, and higher-touch acquisition budget here first.",
            "#0f766e",
        )
    with cols[1]:
        recommendation_card(
            "Retention",
            "Launch a targeted recovery journey",
            f"At-risk customers represent {at_risk_share:.1f}% of the filtered base. Use recency-triggered incentives, service outreach, and replenishment reminders.",
            "#be123c",
        )
    with cols[2]:
        recommendation_card(
            "Category",
            f"Defend demand in {top_category}",
            f"{top_category} contributes the largest filtered spend pool. Package cross-sell recommendations around this category to lift basket value.",
            "#1d4ed8" if loyalty_gap < 8 else "#f59e0b",
        )


def render_core_charts(df: pd.DataFrame, profiles: pd.DataFrame, elbow: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    chart_df = df.copy()
    chart_df["Cluster"] = chart_df["Cluster"].astype(str)

    left, right = st.columns([1, 1])
    with left:
        fig = px.bar(
            profiles,
            x=profiles["Cluster"].astype(str),
            y="Customers",
            color=profiles["Cluster"].astype(str),
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Customer Segment Distribution",
            text="Customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        st.plotly_chart(plot_template(fig), width="stretch")

    with right:
        fig = px.line(elbow, x="Clusters", y="Inertia", markers=True, title="Elbow Method")
        fig.update_traces(line_color="#2563eb", marker=dict(size=9))
        st.plotly_chart(plot_template(fig), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        fig = px.scatter(
            chart_df.sample(min(15000, len(chart_df)), random_state=RANDOM_STATE),
            x="Annual Income",
            y="Spending Score",
            color="Cluster",
            size="Total Spend",
            hover_data=["Customer ID", "Region", "Country", "Preferred Category", "Loyalty Score"],
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Income vs Spending Clusters",
        )
        fig.update_traces(marker=dict(opacity=0.62, line=dict(width=0)))
        st.plotly_chart(plot_template(fig, 430), width="stretch")

    with right:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_data=["Customer ID", "Total Spend", "Loyalty Score", "Region"],
            color_discrete_sequence=COLOR_SEQUENCE,
            title="PCA 2D Cluster Visualization",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.65, line=dict(width=0)))
        st.plotly_chart(plot_template(fig, 430), width="stretch")


def render_advanced_charts(df: pd.DataFrame) -> None:
    chart_df = df.copy()
    chart_df["Cluster"] = chart_df["Cluster"].astype(str)

    st.markdown('<div class="section-title">Behavior and Preference Analysis</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        region_behavior = (
            chart_df.groupby("Region", observed=True)
            .agg(
                Avg_Spend=("Total Spend", "mean"),
                Avg_Loyalty=("Loyalty Score", "mean"),
                Avg_Frequency=("Purchase Frequency", "mean"),
                Customers=("Customer ID", "count"),
            )
            .reset_index()
        )
        fig = px.scatter(
            region_behavior,
            x="Avg_Spend",
            y="Avg_Loyalty",
            size="Customers",
            color="Region",
            hover_data=["Avg_Frequency", "Customers"],
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Region-wise Customer Behavior",
        )
        st.plotly_chart(plot_template(fig), width="stretch")

    with right:
        fig = px.box(
            chart_df,
            x="Cluster",
            y="Loyalty Score",
            color="Cluster",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Loyalty Analysis by Segment",
        )
        st.plotly_chart(plot_template(fig), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        category_counts = (
            chart_df.groupby(["Preferred Category", "Cluster"], observed=True)
            .size()
            .reset_index(name="Customers")
        )
        fig = px.bar(
            category_counts,
            x="Preferred Category",
            y="Customers",
            color="Cluster",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Category Preferences by Segment",
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(plot_template(fig, 430), width="stretch")

    with right:
        fig = px.violin(
            chart_df,
            x="Customer Value",
            y="Total Spend",
            color="Customer Value",
            box=True,
            color_discrete_sequence=["#2563eb", "#dc2626", "#7c3aed", "#16a34a"],
            title="Customer Value Spend Profile",
        )
        st.plotly_chart(plot_template(fig, 430), width="stretch")


def render_segment_table(profiles: pd.DataFrame) -> None:
    display_profiles = profiles.copy()
    currency_cols = ["Avg_Income", "Avg_Spend"]
    percent_cols = ["Share"]
    for col in currency_cols:
        display_profiles[col] = display_profiles[col].map(lambda value: f"${value:,.0f}")
    for col in percent_cols:
        display_profiles[col] = display_profiles[col].map(lambda value: f"{value:.1%}")
    numeric_cols = ["Avg_Age", "Avg_Spending_Score", "Avg_Loyalty", "Avg_Frequency", "Avg_Tenure", "Avg_Recency"]
    for col in numeric_cols:
        display_profiles[col] = display_profiles[col].map(lambda value: f"{value:.1f}")
    st.dataframe(display_profiles, width="stretch", hide_index=True)


def render_download(df: pd.DataFrame) -> None:
    report = df.drop(columns=["Days Since Last Purchase"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered report",
        data=report,
        file_name="filtered_customer_segments.csv",
        mime="text/csv",
        width="stretch",
    )


def main() -> None:
    inject_css()
    render_header()

    with st.spinner("Loading customer data and preparing executive view..."):
        source_df = load_customer_data(DATA_PATH)
    filtered_for_controls, selected_clusters = filter_data(source_df)
    with st.spinner("Training K-Means segmentation model..."):
        labeled_df, profiles, _ = cluster_customers(source_df, selected_clusters)

    # Reapply filters to the clustered table so every view uses the same segment labels.
    filtered_df = labeled_df.loc[filtered_for_controls.index].copy()
    filtered_profiles = (
        filtered_df.groupby("Cluster", observed=True)
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual Income", "mean"),
            Avg_Spend=("Total Spend", "mean"),
            Avg_Spending_Score=("Spending Score", "mean"),
            Avg_Loyalty=("Loyalty Score", "mean"),
            Avg_Frequency=("Purchase Frequency", "mean"),
            Avg_Tenure=("Customer Tenure", "mean"),
            Avg_Recency=("Days Since Last Purchase", "mean"),
        )
        .reset_index()
    )
    if not filtered_profiles.empty:
        filtered_profiles["Share"] = filtered_profiles["Customers"] / len(filtered_df)

    render_kpis(filtered_df, selected_clusters)
    render_business_insights(filtered_df)
    render_strategic_recommendations(filtered_df)

    if filtered_df.empty:
        return

    with st.spinner("Rendering model diagnostics and PCA projection..."):
        elbow = compute_elbow(source_df)
        pca_df = compute_pca(filtered_df)

    st.markdown('<div class="section-title">Segmentation Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Model diagnostics and visual cluster separation for executive review.</div>', unsafe_allow_html=True)
    render_core_charts(filtered_df, filtered_profiles, elbow, pca_df)
    render_advanced_charts(filtered_df)

    st.markdown('<div class="section-title">Segment Profiles</div>', unsafe_allow_html=True)
    render_segment_table(filtered_profiles)

    st.markdown('<div class="section-title">Filtered Customer Report</div>', unsafe_allow_html=True)
    render_download(filtered_df)
    st.dataframe(filtered_df.head(1000), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
