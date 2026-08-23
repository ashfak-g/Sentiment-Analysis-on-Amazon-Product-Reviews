"""
Model evaluation module for Amazon Product Review Sentiment Analysis.
Computes evaluation metrics and generates visualization plots.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from src.utils import logger, ensure_directories

class Evaluator:
    """Evaluates classification models and generates graphical reports."""

    @staticmethod
    def evaluate_model(model: Any, X_test: Any, y_test: Any, model_name: str = "Model") -> Dict[str, Any]:
        """Calculates evaluation metrics for a given model and test set."""
        y_pred = model.predict(X_test)
        
        # Calculate probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                y_proba = model.decision_function(X_test)
            except Exception:
                pass

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
        cm = confusion_matrix(y_test, y_pred).tolist()

        results = {
            "model_name": model_name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc) if not np.isnan(auc) else None,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        logger.info(
            f"Results for [{model_name}] -> "
            f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc if not np.isnan(auc) else 'N/A'}"
        )
        return results

    @staticmethod
    def compare_models(results_list: list[Dict[str, Any]]) -> pd.DataFrame:
        """Compiles evaluation results into a pandas comparison DataFrame."""
        summary = []
        for r in results_list:
            summary.append({
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1 Score": r["f1_score"],
                "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else np.nan
            })
        df_comp = pd.DataFrame(summary).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
        return df_comp
