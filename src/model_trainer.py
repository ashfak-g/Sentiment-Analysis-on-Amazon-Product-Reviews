"""
Model training module for Amazon Product Review Sentiment Analysis.
"""

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.utils import logger

class ModelTrainer:
    """Factory and trainer for classification algorithms."""

    SUPPORTED_MODELS = [
        "logistic_regression",
        "naive_bayes",
        "svm",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm"
    ]

    @staticmethod
    def get_model(model_name: str, **kwargs) -> Any:
        """Instantiates classification model based on name and parameters."""
        name = model_name.lower()
        
        if name == "logistic_regression":
            params = {"max_iter": 1000, "random_state": 42, "C": 1.0, "class_weight": "balanced"}
            params.update(kwargs)
            return LogisticRegression(**params)
            
        elif name == "naive_bayes":
            params = {"alpha": 1.0}
            params.update(kwargs)
            return MultinomialNB(**params)
            
        elif name == "svm":
            # Fast LinearSVC for high dimensional text matrices
            params = {"random_state": 42, "C": 1.0}
            params.update(kwargs)
            return LinearSVC(**params)
                
        elif name == "random_forest":
            params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
            params.update(kwargs)
            return RandomForestClassifier(**params)
            
        elif name == "gradient_boosting":
            params = {"n_estimators": 100, "random_state": 42}
            params.update(kwargs)
            return GradientBoostingClassifier(**params)
            
        elif name == "xgboost":
            if xgb is None:
                raise ImportError("XGBoost library is not installed.")
            params = {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss", "use_label_encoder": False}
            params.update(kwargs)
            return xgb.XGBClassifier(**params)
            
        elif name == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM library is not installed.")
            params = {"n_estimators": 100, "random_state": 42, "verbosity": -1}
            params.update(kwargs)
            return lgb.LGBMClassifier(**params)
            
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from: {ModelTrainer.SUPPORTED_MODELS}")

    @classmethod
    def train_model(cls, model_name: str, X_train: Any, y_train: Any, **kwargs) -> Tuple[Any, float]:
        """Trains specified model and returns trained model object and training duration."""
        import time
        model = cls.get_model(model_name, **kwargs)
        logger.info(f"Starting training for {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        logger.info(f"Completed training for {model_name} in {duration:.2f} seconds.")
        return model, duration
