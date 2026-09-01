"""
Unit tests for model explainability module.
"""

import pytest
from src.feature_extraction import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.text_preprocessor import TextPreprocessor
from src.explainability import ModelExplainability

@pytest.fixture
def trained_components():
    corpus = [
        "Great product fast shipping excellent quality",
        "Terrible quality broken item worst purchase ever"
    ]
    labels = [1, 0]
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.transform(corpus)
    
    vectorizer = FeatureExtractor(max_features=20, min_df=1)
    X = vectorizer.fit_transform(cleaned)
    
    model, _ = ModelTrainer.train_model("logistic_regression", X, labels)
    return vectorizer, model, preprocessor

def test_explainability_top_features(trained_components):
    vectorizer, model, _ = trained_components
    explainer = ModelExplainability(vectorizer, model)
    
    df_pos, df_neg = explainer.get_top_features(top_n=2)
    assert len(df_pos) == 2
    assert len(df_neg) == 2
    assert "feature" in df_pos.columns
    assert "coefficient" in df_pos.columns

def test_explain_text_prediction(trained_components):
    vectorizer, model, preprocessor = trained_components
    explainer = ModelExplainability(vectorizer, model)
    
    explanation = explainer.explain_text_prediction(preprocessor, "Great product but broken item")
    assert "token_contributions" in explanation
    assert len(explanation["token_contributions"]) > 0
    assert "intercept" in explanation
