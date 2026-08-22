"""
Hyperparameter tuning module for Amazon Product Review Sentiment Analysis.
"""

from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.model_trainer import ModelTrainer
from src.utils import logger

class HyperparameterTuner:
    """Automates model hyperparameter tuning using Grid or Randomized Search."""

    PARAM_GRIDS = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        },
        "naive_bayes": {
            "alpha": [0.1, 0.5, 1.0, 2.0]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "xgboost": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9]
        }
    }

    def __init__(self, search_type: str = "grid", cv: int = 5, scoring: str = "f1"):
        self.search_type = search_type.lower()
        self.cv = cv
        self.scoring = scoring

    def tune(
        self, 
        model_name: str, 
        X_train: Any, 
        y_train: Any, 
        custom_param_grid: Dict[str, list] = None,
        n_iter: int = 10
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Runs search cross-validation and returns best estimator, best params, and best score."""
        base_model = ModelTrainer.get_model(model_name)
        param_grid = custom_param_grid or self.PARAM_GRIDS.get(model_name, {})
        
        if not param_grid:
            logger.warning(f"No param grid found for {model_name}. Returning base model.")
            base_model.fit(X_train, y_train)
            return base_model, {}, 0.0

        logger.info(f"Tuning {model_name} using {self.search_type.upper()} Search (CV={self.cv}, metric={self.scoring})...")

        if self.search_type == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1
            )
        elif self.search_type == "random":
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Invalid search type: {self.search_type}")

        search.fit(X_train, y_train)
        
        logger.info(f"Best params for {model_name}: {search.best_params_}")
        logger.info(f"Best CV {self.scoring} score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
