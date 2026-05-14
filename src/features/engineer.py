"""
src/features/engineer.py

Feature engineering functions for churn prediction.
Each function adds one or more business-meaningful features.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_clv_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Lifetime Value (CLV) estimate.
    Simple proxy: MonthlyCharges × tenure.
    A customer paying $80/mo for 24 months has CLV ≈ $1,920.
    """
    df = df.copy()
    df["clv_estimate"] = df["MonthlyCharges"] * df["tenure"]
    logger.info("Added: clv_estimate")
    return df


def add_services_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of add-on services subscribed.
    Hypothesis: customers with more services are more sticky.
    """
    df = df.copy()
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    # Handle already-encoded (1/0) or raw Yes/No columns
    available = [c for c in service_cols if c in df.columns]
    if all(df[available].dtypes == "int64") or all(df[available].dtypes == "float64"):
        df["services_count"] = df[available].sum(axis=1)
    else:
        df["services_count"] = df[available].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )
    logger.info(f"Added: services_count (from {len(available)} service cols)")
    return df


def add_contract_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags for contract type.
    Month-to-month customers churn at significantly higher rates.
    """
    df = df.copy()
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["is_long_term"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)
        logger.info("Added: is_month_to_month, is_long_term")
    return df


def add_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly spend per subscribed service.
    High charge per service may indicate low perceived value.
    """
    df = df.copy()
    if "services_count" not in df.columns:
        df = add_services_count(df)
    df["charge_per_service"] = df["MonthlyCharges"] / df["services_count"].replace(0, 1)
    logger.info("Added: charge_per_service")
    return df


def add_tenure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure into lifecycle stages.
    New customers (< 12 months) churn at higher rates.
    """
    df = df.copy()
    df["tenure_segment"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, float("inf")],
        labels=["new", "developing", "established", "loyal"],
    )
    logger.info("Added: tenure_segment")
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering functions in sequence.
    Call this as the single entry point for feature creation.

    Args:
        df: Cleaned, encoded DataFrame

    Returns:
        DataFrame with all engineered features added
    """
    logger.info("Building engineered feature set...")
    df = add_clv_estimate(df)
    df = add_services_count(df)
    df = add_contract_flags(df)
    df = add_charge_per_service(df)
    df = add_tenure_bins(df)
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
