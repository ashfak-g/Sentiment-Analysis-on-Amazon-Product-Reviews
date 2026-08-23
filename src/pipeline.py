"""
End-to-End Orchestrator Pipeline for Amazon Product Review Sentiment Analysis.
Executes data ingestion, text preprocessing, feature extraction, multi-model training,
hyperparameter tuning, evaluation, and model artifact serialization.
"""

import os
import time
from typing import Dict, Any, Tuple
import pandas as pd

from src.utils import logger, ensure_directories, save_artifact, save_json
from src.data_loader import DataLoader
from src.text_preprocessor import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.hyperparameter_tuner import HyperparameterTuner
from src.evaluator import Evaluator

MODEL_DIR = os.path.join("models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

class SentimentPipeline:
    """Production Pipeline Orchestrator."""

    def __init__(self, models_to_run: list[str] = None, tune_best: bool = True):
        self.models_to_run = models_to_run or [
            "logistic_regression",
            "naive_bayes",
            "svm",
            "random_forest",
            "xgboost"
        ]
        self.tune_best = tune_best
        ensure_directories([MODEL_DIR, "data/raw", "data/processed"])

    def run(self) -> Tuple[Any, FeatureExtractor, TextPreprocessor, pd.DataFrame]:
        """Runs full training, tuning, evaluation, and saving workflow."""
        logger.info("================ STARTING END-TO-END ML PIPELINE ================")
        
        # 1. Load Data
        loader = DataLoader()
        raw_df = loader.load_raw_data()
        train_df, test_df = loader.split_and_save(raw_df)
        
        # 2. Text Preprocessing
        logger.info("Preprocessing text data...")
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        train_cleaned = preprocessor.transform(train_df['reviewText'].tolist())
        test_cleaned = preprocessor.transform(test_df['reviewText'].tolist())
        
        y_train = train_df['Positive'].values
        y_test = test_df['Positive'].values
        
        # 3. Feature Extraction
        logger.info("Extracting TF-IDF features...")
        vectorizer = FeatureExtractor(vectorizer_type="tfidf", max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_cleaned)
        X_test = vectorizer.transform(test_cleaned)
        
        # 4. Multi-Model Baseline Training & Evaluation
        evaluation_results = []
        trained_models = {}
        
        for m_name in self.models_to_run:
            try:
                model, duration = ModelTrainer.train_model(m_name, X_train, y_train)
                res = Evaluator.evaluate_model(model, X_test, y_test, model_name=m_name)
                res["training_duration_sec"] = duration
                evaluation_results.append(res)
                trained_models[m_name] = model
            except Exception as e:
                logger.error(f"Error training model '{m_name}': {e}")
                
        comparison_df = Evaluator.compare_models(evaluation_results)
        logger.info("\n--- Baseline Model Comparison ---\n" + comparison_df.to_string(index=False))
        
        # Determine initial top model
        top_model_name = comparison_df.iloc[0]["Model"]
        best_model = trained_models[top_model_name]
        best_score = comparison_df.iloc[0]["F1 Score"]
        
        # 5. Hyperparameter Tuning on Top Model
        if self.tune_best:
            logger.info(f"Tuning hyper-parameters for top model: {top_model_name}")
            tuner = HyperparameterTuner(search_type="grid", cv=5, scoring="f1")
            tuned_model, best_params, tuned_score = tuner.tune(top_model_name, X_train, y_train)
            
            # Evaluate tuned model
            tuned_res = Evaluator.evaluate_model(tuned_model, X_test, y_test, model_name=f"{top_model_name}_tuned")
            if tuned_res["f1_score"] >= best_score:
                logger.info(f"Tuned model improved score from {best_score:.4f} to {tuned_res['f1_score']:.4f}")
                best_model = tuned_model
                top_model_name = f"{top_model_name}_tuned"
                evaluation_results.append(tuned_res)
                comparison_df = Evaluator.compare_models(evaluation_results)

        # 6. Save Artifacts
        logger.info("Saving best model pipeline artifacts...")
        save_artifact(best_model, BEST_MODEL_PATH)
        save_artifact(vectorizer.vectorizer, VECTORIZER_PATH)
        save_artifact(preprocessor, PREPROCESSOR_PATH)
        
        best_metrics = [r for r in evaluation_results if r["model_name"] == top_model_name][0]
        metadata = {
            "best_model_name": top_model_name,
            "metrics": {
                "accuracy": best_metrics["accuracy"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1_score": best_metrics["f1_score"],
                "roc_auc": best_metrics["roc_auc"]
            },
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "num_features": X_train.shape[1],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_json(metadata, METADATA_PATH)
        
        logger.info("================ PIPELINE COMPLETED SUCCESSFULLY ================")
        return best_model, vectorizer, preprocessor, comparison_df

if __name__ == "__main__":
    pipeline = SentimentPipeline()
    pipeline.run()
