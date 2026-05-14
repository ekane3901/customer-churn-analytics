"""
src/dashboard/app.py

Streamlit dashboard for the Customer Churn & Revenue Risk Analytics Platform.
Run with: streamlit run src/dashboard/app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn & Revenue Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions(path: str = "data/processed/predictions.csv") -> pd.DataFrame:
    """Load pre-computed risk scores. Falls back to synthetic demo data."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning("predictions.csv not found — showing synthetic demo data.")
        return _generate_demo_data()


def _generate_demo_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate plausible synthetic data for UI demo purposes."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    churn_prob = rng.beta(2, 5, n).round(4)
    clv = (monthly * tenure).round(2)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "churn_probability": churn_prob,
        "predicted_churn": (churn_prob >= 0.5).astype(int),
        "clv_estimate": clv,
        "revenue_risk_score": (churn_prob * clv).round(2),
        "tenure": tenure,
        "monthly_charges": monthly,
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "internet_service": rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22]),
        "tenure_segment": pd.cut(
            tenure, bins=[0, 12, 24, 48, 72],
            labels=["new", "developing", "established", "loyal"]
        ).astype(str),
    })


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Filters")
risk_threshold = st.sidebar.slider("Churn Probability Threshold", 0.0, 1.0, 0.5, 0.05)
top_n = st.sidebar.slider("Top N At-Risk Customers", 10, 100, 25)
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    default=["Month-to-month", "One year", "Two year"],
)

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_predictions()
df_filtered = df[df["contract"].isin(contract_filter)] if "contract" in df.columns else df

at_risk = df_filtered[df_filtered["churn_probability"] >= risk_threshold]
top_risk = at_risk.nlargest(top_n, "revenue_risk_score")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn & Revenue Risk Dashboard")
st.caption("Powered by XGBoost · Revenue Risk = Churn Probability × Customer Lifetime Value")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df_filtered):,}")
col2.metric("Predicted Churners", f"{len(at_risk):,}", f"{len(at_risk)/len(df_filtered):.1%} churn rate")
col3.metric("Total Revenue at Risk", f"${at_risk['revenue_risk_score'].sum():,.0f}")
col4.metric("Avg Churn Probability", f"{at_risk['churn_probability'].mean():.1%}")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(
        df_filtered, x="churn_probability", nbins=40,
        color_discrete_sequence=["#e74c3c"],
        labels={"churn_probability": "Churn Probability"},
    )
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold ({risk_threshold})")
    fig.update_layout(bargap=0.05, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Revenue at Risk by Contract Type")
    if "contract" in df_filtered.columns:
        risk_by_contract = (
            at_risk.groupby("contract")["revenue_risk_score"]
            .sum()
            .reset_index()
            .sort_values("revenue_risk_score", ascending=True)
        )
        fig2 = px.bar(
            risk_by_contract, x="revenue_risk_score", y="contract",
            orientation="h", color_discrete_sequence=["#2980b9"],
            labels={"revenue_risk_score": "Total Revenue at Risk ($)", "contract": "Contract Type"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ── Scatter: CLV vs Churn Probability ─────────────────────────────────────────
st.subheader("Revenue Risk Landscape — CLV vs. Churn Probability")
st.caption("Top-right quadrant: high-value customers most likely to churn — highest priority for retention.")
fig3 = px.scatter(
    df_filtered.sample(min(500, len(df_filtered))),
    x="churn_probability", y="clv_estimate",
    color="revenue_risk_score",
    color_continuous_scale="Reds",
    size="revenue_risk_score",
    size_max=18,
    labels={
        "churn_probability": "Predicted Churn Probability",
        "clv_estimate": "Estimated Customer Lifetime Value ($)",
        "revenue_risk_score": "Revenue Risk Score",
    },
    hover_data=["customer_id", "tenure", "monthly_charges"] if "customer_id" in df_filtered.columns else None,
)
fig3.add_vline(x=risk_threshold, line_dash="dash", line_color="gray", opacity=0.6)
fig3.update_layout(height=420)
st.plotly_chart(fig3, use_container_width=True)

# ── At-Risk Customer Table ─────────────────────────────────────────────────────
st.subheader(f"🚨 Top {top_n} Customers by Revenue Risk Score")
st.caption("Sorted by revenue impact — prioritize retention outreach from the top down.")

display_cols = [c for c in [
    "customer_id", "churn_probability", "revenue_risk_score",
    "clv_estimate", "monthly_charges", "tenure", "contract"
] if c in top_risk.columns]

st.dataframe(
    top_risk[display_cols]
    .style
    .format({
        "churn_probability": "{:.1%}",
        "revenue_risk_score": "${:,.2f}",
        "clv_estimate": "${:,.2f}",
        "monthly_charges": "${:.2f}",
    })
    .background_gradient(subset=["revenue_risk_score"], cmap="Reds"),
    use_container_width=True,
    height=420,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Customer Churn & Revenue Risk Analytics Platform · Built with Python, XGBoost, AWS, and Streamlit")
