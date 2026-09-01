"""
Model Explainability & Feature Importance Analysis Module.
Extracts top positive and negative word coefficients and explains individual review predictions.
"""

import os
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for CI/CD server compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import logger, ensure_directories

class ModelExplainability:
    """Provides global feature importance and local text prediction explanations."""

    def __init__(self, vectorizer: Any, model: Any):
        # Unwrap FeatureExtractor if needed
        self.vectorizer = getattr(vectorizer, "vectorizer", vectorizer)
        self.model = model
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Extract coefficients for linear models (Logistic Regression, LinearSVC)
        if hasattr(self.model, "coef_"):
            self.coefs = np.ravel(self.model.coef_)
        else:
            raise ValueError("Model does not have 'coef_' attribute. Model explainability currently supports linear models.")

    def get_top_features(self, top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns top N positive and top N negative features with their weight coefficients.
        """
        num_features = len(self.coefs)
        n = min(top_n, max(1, num_features // 2)) if num_features > 0 else top_n
        
        sorted_indices = np.argsort(self.coefs)
        
        top_neg_indices = sorted_indices[:n]
        top_pos_indices = sorted_indices[-n:][::-1]
        
        df_pos = pd.DataFrame({
            "feature": self.feature_names[top_pos_indices],
            "coefficient": self.coefs[top_pos_indices],
            "impact": "Positive"
        })
        
        df_neg = pd.DataFrame({
            "feature": self.feature_names[top_neg_indices],
            "coefficient": self.coefs[top_neg_indices],
            "impact": "Negative"
        })
        
        logger.info(f"Extracted Top {len(df_pos)} positive and negative features.")
        return df_pos, df_neg

    def explain_text_prediction(self, preprocessor: Any, text: str) -> Dict[str, Any]:
        """
        Explains a specific review text by showing token-level coefficient contributions.
        """
        cleaned_text = preprocessor.clean_text(text)
        tokens = cleaned_text.split()
        vocab = self.vectorizer.vocabulary_
        
        token_contributions = []
        intercept = float(np.ravel(self.model.intercept_)[0]) if hasattr(self.model, "intercept_") else 0.0
        
        total_score = intercept
        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                weight = float(self.coefs[idx])
                total_score += weight
                token_contributions.append({
                    "token": token,
                    "weight": weight,
                    "impact": "Positive" if weight > 0 else "Negative"
                })
            else:
                token_contributions.append({
                    "token": token,
                    "weight": 0.0,
                    "impact": "Out of Vocabulary"
                })
                
        # Sort tokens by magnitude of impact
        token_contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
        
        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "intercept": intercept,
            "total_linear_score": total_score,
            "token_contributions": token_contributions
        }

    def plot_top_features(self, top_n: int = 15, save_path: str = None) -> plt.Figure:
        """
        Plots a horizontal bar chart of the top N positive and negative feature weights.
        """
        df_pos, df_neg = self.get_top_features(top_n=top_n)
        df_combined = pd.concat([df_pos, df_neg]).sort_values(by="coefficient")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#e74c3c" if val < 0 else "#2ecc71" for val in df_combined["coefficient"]]
        
        sns.barplot(
            x="coefficient",
            y="feature",
            data=df_combined,
            hue="feature",
            palette=colors,
            legend=False,
            ax=ax
        )
        
        ax.set_title(f"Top {len(df_pos)} Positive and Negative Sentiment Features", fontsize=14, fontweight="bold")
        ax.set_xlabel("TF-IDF Model Coefficient (Impact Weight)", fontsize=12)
        ax.set_ylabel("Word / N-Gram Feature", fontsize=12)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            ensure_directories([os.path.dirname(save_path)])
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")
            
        return fig
