"""
Feature extraction module for Amazon Product Review Sentiment Analysis.
"""

from typing import Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.utils import logger, save_artifact, load_artifact

class FeatureExtractor:
    """Wrapper for TF-IDF and Count vectorizers with persistence capabilities."""

    def __init__(
        self, 
        vectorizer_type: str = "tfidf", 
        max_features: int = 5000, 
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2
    ):
        self.vectorizer_type = vectorizer_type.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        
        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=True
            )
        elif self.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df
            )
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

    def fit_transform(self, raw_documents: list[str]) -> Any:
        """Fits the vectorizer and transforms documents into feature matrix."""
        logger.info(f"Fitting {self.vectorizer_type.upper()} vectorizer (max_features={self.max_features})...")
        X = self.vectorizer.fit_transform(raw_documents)
        logger.info(f"Feature matrix shape: {X.shape}")
        return X

    def transform(self, raw_documents: list[str]) -> Any:
        """Transforms documents into feature matrix using fitted vectorizer."""
        return self.vectorizer.transform(raw_documents)

    def get_feature_names_out(self) -> np.ndarray:
        """Delegates feature names retrieval to underlying vectorizer."""
        return self.vectorizer.get_feature_names_out()

    @property
    def vocabulary_(self) -> dict:
        """Delegates vocabulary retrieval to underlying vectorizer."""
        return self.vectorizer.vocabulary_

    def save(self, filepath: str) -> str:
        """Saves fitted vectorizer artifact."""
        return save_artifact(self.vectorizer, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeatureExtractor":
        """Loads fitted vectorizer artifact."""
        instance = cls()
        instance.vectorizer = load_artifact(filepath)
        return instance
