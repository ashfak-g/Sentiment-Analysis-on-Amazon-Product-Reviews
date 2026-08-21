"""
Data loading and splitting module for Amazon Product Review Sentiment Analysis.
"""

import os
import urllib.request
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.utils import logger, ensure_directories

DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "amazon.csv")
PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "train.csv")
PROCESSED_TEST_PATH = os.path.join("data", "processed", "test.csv")

class DataLoader:
    """Handles dataset fetching, loading, cleaning, and train/test splitting."""

    def __init__(self, raw_path: str = RAW_DATA_PATH, url: str = DEFAULT_DATASET_URL):
        self.raw_path = raw_path
        self.url = url
        ensure_directories([os.path.dirname(self.raw_path), "data/processed"])

    def download_if_missing(self) -> str:
        """Downloads dataset from URL if it does not exist locally."""
        if not os.path.exists(self.raw_path):
            logger.info(f"Downloading dataset from {self.url} to {self.raw_path}...")
            try:
                urllib.request.urlretrieve(self.url, self.raw_path)
                logger.info("Download completed successfully.")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                raise
        else:
            logger.info(f"Raw dataset found at {self.raw_path}")
        return self.raw_path

    def load_raw_data(self) -> pd.DataFrame:
        """Loads the raw dataset into a pandas DataFrame."""
        filepath = self.download_if_missing()
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        logger.info(f"Loaded raw dataset with shape: {df.shape}")
        
        # Verify required columns
        required_cols = {'reviewText', 'Positive'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")
            
        return df

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes null or empty reviewText entries."""
        initial_count = len(df)
        df = df.dropna(subset=['reviewText', 'Positive']).copy()
        df['reviewText'] = df['reviewText'].astype(str).str.strip()
        df = df[df['reviewText'] != '']
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing or blank reviewText.")
        return df

    def split_and_save(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits cleaned dataframe into train and test sets and saves to disk."""
        df_clean = self.clean_missing_values(df)
        
        train_df, test_df = train_test_split(
            df_clean,
            test_size=test_size,
            random_state=random_state,
            stratify=df_clean['Positive']
        )
        
        train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(PROCESSED_TEST_PATH, index=False)
        
        logger.info(f"Train split saved: {train_df.shape} -> {PROCESSED_TRAIN_PATH}")
        logger.info(f"Test split saved: {test_df.shape} -> {PROCESSED_TEST_PATH}")
        
        return train_df, test_df

if __name__ == "__main__":
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    train_df, test_df = loader.split_and_save(raw_df)
