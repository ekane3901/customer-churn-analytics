"""
src/models/train.py

Train logistic regression and XGBoost churn classifiers.
Saves models and logs evaluation metrics.
"""

import os
import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.ingest import load_data
from src.data.preprocess import clean, encode_categoricals
from src.features.engineer import build_feature_set
from src.models.evaluate import print_evaluation_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_features(df: pd.DataFrame, config: dict):
    """
    Split DataFrame into feature matrix X and target vector y.
    Drops non-numeric and engineered label columns.
    """
    target = config["data"]["target_column"]
    drop_cols = [target, "tenure_segment"]  # drop categoricals not yet encoded
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]

    X = df[feature_cols]
    y = df[target]
    logger.info(f"Features: {len(feature_cols)} | Samples: {len(y)} | Churn rate: {y.mean():.1%}")
    return X, y


def train_logistic_regression(X_train, y_train, config: dict) -> Pipeline:
    """Logistic regression with standard scaling."""
    params = config["models"]["logistic_regression"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])
    pipeline.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return pipeline


def train_xgboost(X_train, y_train, X_val, y_val, config: dict) -> XGBClassifier:
    """XGBoost with early stopping on validation set."""
    params = config["models"]["xgboost"].copy()
    params.pop("random_state", None)

    model = XGBClassifier(
        **params,
        random_state=config["data"]["random_state"],
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"XGBoost trained. Best iteration: {model.best_iteration}")
    return model


def save_model(model, name: str, output_dir: str = "models/") -> str:
    """Serialize model to disk with joblib."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def main():
    config = load_config()

    # ── Load & prepare data ────────────────────────────────────────────
    df_raw = load_data(use_s3=False)
    df_clean = clean(df_raw)
    df_encoded = encode_categoricals(df_clean)
    df_features = build_feature_set(df_encoded)

    X, y = prepare_features(df_features, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    # Further split train into train/val for XGBoost early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # ── Train models ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_tr, y_tr, config)
    print_evaluation_report(lr_model, X_test, y_test, model_name="Logistic Regression")
    save_model(lr_model, "logistic_regression")

    logger.info("=" * 50)
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, config)
    print_evaluation_report(xgb_model, X_test, y_test, model_name="XGBoost")
    save_model(xgb_model, "xgboost")

    logger.info("=" * 50)
    logger.info("Training complete. Models saved to models/")

    return lr_model, xgb_model, X_test, y_test


if __name__ == "__main__":
    main()
