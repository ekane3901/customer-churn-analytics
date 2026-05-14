"""
src/data/preprocess.py

Clean and type-cast raw Telco churn data.
All transformations are logged for reproducibility.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for raw Telco churn CSV.

    Steps:
      1. Fix TotalCharges (stored as string, spaces for new customers)
      2. Encode target variable: Churn Yes/No -> 1/0
      3. Drop customerID from features (keep as index)
      4. Strip whitespace from string columns
      5. Log summary of missing values

    Args:
        df: Raw DataFrame from ingest module

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    logger.info("Starting data cleaning pipeline...")

    # 1. Fix TotalCharges — new customers have blank strings
    logger.info("Casting TotalCharges to numeric (blank -> NaN -> fill with MonthlyCharges)")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # New customers (tenure=0) have no total charges yet; fill with MonthlyCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]
    logger.info(f"  Fixed {mask.sum()} TotalCharges NaN values")

    # 2. Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    logger.info(f"  Churn rate: {df['Churn'].mean():.1%}")

    # 3. Set customerID as index
    if "customerID" in df.columns:
        df = df.set_index("customerID")

    # 4. Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 5. Missing value report
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values detected:\n" + str(missing[missing > 0]))
    else:
        logger.info("No missing values after cleaning.")

    logger.info(f"Cleaning complete. Shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    Binary Yes/No columns are label-encoded instead.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with encoded features
    """
    df = df.copy()

    # Binary columns: map Yes -> 1, No -> 0
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # Gender: Female -> 1, Male -> 0
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # Multi-class categoricals: one-hot encode
    ohe_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=False)

    logger.info(f"Encoding complete. Shape after encoding: {df.shape}")
    return df
