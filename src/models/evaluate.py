"""
src/models/evaluate.py

Evaluation metrics and reporting for churn classifiers.
Uses business-relevant metrics, not just accuracy.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


def print_evaluation_report(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Print a full evaluation report for a binary classifier.

    Returns dict of key metrics for logging/comparison.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(f"  AUC-ROC:            {auc:.4f}")
    print(f"  Avg Precision (AP): {ap:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay', 'Churn'])}")

    return {"auc": auc, "ap": ap, "f1": f1, "precision": precision, "recall": recall}


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        models: dict of {name: fitted_model}
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Classifiers")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20, save_path: str = None):
    """
    Plot XGBoost feature importances.
    Works with XGBClassifier and sklearn Pipeline.
    """
    if hasattr(model, "named_steps"):  # Pipeline
        clf = model.named_steps.get("clf", model)
    else:
        clf = model

    if not hasattr(clf, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute.")
        return None

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")
    return fig


def compute_revenue_risk(df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
    """
    Attach churn probabilities and revenue risk score to customer DataFrame.

    Revenue risk = churn_probability × estimated CLV
    This ranks customers by business impact, not just likelihood to churn.

    Args:
        df: DataFrame with 'clv_estimate' column
        churn_probs: Array of predicted churn probabilities

    Returns:
        DataFrame sorted by revenue_risk_score descending
    """
    result = df.copy()
    result["churn_probability"] = churn_probs

    if "clv_estimate" in result.columns:
        result["revenue_risk_score"] = result["churn_probability"] * result["clv_estimate"]
    else:
        logger.warning("clv_estimate not found; revenue_risk_score = churn_probability only")
        result["revenue_risk_score"] = result["churn_probability"]

    result = result.sort_values("revenue_risk_score", ascending=False)
    return result
