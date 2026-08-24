"""
Unit tests for feature extraction and model training modules.
"""

import pytest
from src.feature_extraction import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator

def test_feature_extractor():
    corpus = ["Great product fast shipping", "Terrible quality broken item"]
    fe = FeatureExtractor(vectorizer_type="tfidf", max_features=10, min_df=1)
    X = fe.fit_transform(corpus)
    assert X.shape[0] == 2
    assert X.shape[1] > 0

def test_model_trainer():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    y = [1, 0, 1, 0]
    model, duration = ModelTrainer.train_model("logistic_regression", X, y)
    assert model is not None
    assert duration >= 0.0
    
    res = Evaluator.evaluate_model(model, X, y, model_name="TestLR")
    assert "accuracy" in res
    assert res["accuracy"] == 1.0
